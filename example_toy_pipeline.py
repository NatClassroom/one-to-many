"""
Complete One-to-Many Pipeline: Toy Example

This self-contained file shows the ENTIRE one-to-many pipeline on a tiny
dataset so you can see every piece before tackling the subtasks.

Dataset: 12 names across 2 categories
  - "cat" names: Whiskers, Mittens, Shadow, Pepper, Luna, Felix
  - "dog" names: Buddy, Rocky, Tucker, Buster, Duke, Bruno

The model learns that cat names tend to have more letters like 'i', 'e', 's'
while dog names have more 'u', 'k', 'o' sounds.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Step 1: Define vocabulary and dataset
# =============================================================================

PAD, SOS, EOS = 0, 1, 2
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>"]
CHARS = list("abcdefghijklmnopqrstuvwxyz")
ALL_TOKENS = SPECIAL_TOKENS + CHARS

char_to_idx = {ch: i for i, ch in enumerate(ALL_TOKENS)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
VOCAB_SIZE = len(ALL_TOKENS)  # 29

CATEGORIES = ["cat", "dog"]
NAMES = {
    "cat": ["whiskers", "mittens", "shadow", "pepper", "luna", "felix"],
    "dog": ["buddy", "rocky", "tucker", "buster", "duke", "bruno"],
}


def encode_name(name):
    """Convert a name string to [SOS, c1, c2, ..., EOS] tensor."""
    return torch.tensor([SOS] + [char_to_idx[c] for c in name] + [EOS])


def make_batch():
    """Create a single batch of all 12 names with padding."""
    categories = []
    sequences = []
    for cat_idx, cat_name in enumerate(CATEGORIES):
        for name in NAMES[cat_name]:
            categories.append(cat_idx)
            sequences.append(encode_name(name))

    # Pad to max length
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(sequences), max_len), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    return torch.tensor(categories), padded


# =============================================================================
# Step 2: Define the model (all in one class)
# =============================================================================


class ToyOneToMany(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_categories=2, embed_dim=16, hidden_size=32):
        super().__init__()
        # Encoder: category -> initial hidden state
        self.cat_emb = nn.Embedding(num_categories, embed_dim)
        self.cat_to_h = nn.Linear(embed_dim, hidden_size)
        self.cat_to_c = nn.Linear(embed_dim, hidden_size)

        # Decoder: character-level LSTM
        self.char_emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def encode(self, categories):
        """Category -> initial (h, c)."""
        emb = self.cat_emb(categories)
        return torch.tanh(self.cat_to_h(emb)), torch.tanh(self.cat_to_c(emb))

    def forward_teacher_forcing(self, categories, targets):
        """
        Training with teacher forcing.

        targets: [SOS, c1, c2, ..., cn, EOS, PAD, ...]

        At each step, we feed the GROUND TRUTH character as input
        (not the model's prediction). This makes training stable.
        """
        h, c = self.encode(categories)

        # Input = everything except last token (we don't predict from EOS/PAD)
        # Target = everything except first token (we don't predict SOS)
        input_seq = targets[:, :-1]  # [SOS, c1, ..., cn]
        num_steps = input_seq.shape[1]

        all_logits = []
        for t in range(num_steps):
            emb = self.char_emb(input_seq[:, t])  # Always ground truth!
            h, c = self.lstm_cell(emb, (h, c))
            logits = self.output(h)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # (batch, num_steps, vocab)

    def generate(self, category_idx, max_len=15, temperature=0.7):
        """Generate a name by sampling one character at a time."""
        self.eval()
        with torch.no_grad():
            h, c = self.encode(torch.tensor([category_idx]))
            current = torch.tensor([SOS])
            chars = []
            for _ in range(max_len):
                emb = self.char_emb(current)
                h, c = self.lstm_cell(emb, (h, c))
                logits = self.output(h) / temperature
                probs = F.softmax(logits, dim=-1)
                current = torch.multinomial(probs, 1).squeeze(1)
                if current.item() == EOS:
                    break
                chars.append(idx_to_char[current.item()])
        self.train()
        return "".join(chars)


# =============================================================================
# Step 3: Train
# =============================================================================


def train():
    torch.manual_seed(42)
    random.seed(42)

    model = ToyOneToMany()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    categories, targets = make_batch()

    print("=" * 50)
    print("Toy One-to-Many Pipeline: Teacher Forcing")
    print("=" * 50)
    print(f"\nDataset: {sum(len(v) for v in NAMES.values())} names, 2 categories")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"\nTraining data:")
    for cat in CATEGORIES:
        print(f"  {cat}: {', '.join(NAMES[cat])}")

    print(f"\nTraining for 200 epochs...\n")

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        # Forward with teacher forcing
        logits = model.forward_teacher_forcing(categories, targets)

        # Loss: compare predictions to target (targets shifted by 1)
        target_seq = targets[:, 1:]  # [c1, c2, ..., EOS, PAD]
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")
            for cat_idx, cat_name in enumerate(CATEGORIES):
                names = [model.generate(cat_idx) for _ in range(3)]
                print(f"    {cat_name}: {', '.join(names)}")
            print()

    # Final showcase
    print("=" * 50)
    print("Final generation (temperature = 0.5)")
    print("=" * 50)
    for cat_idx, cat_name in enumerate(CATEGORIES):
        names = [model.generate(cat_idx, temperature=0.5) for _ in range(5)]
        print(f"  {cat_name}: {', '.join(names)}")

    print(f"""
Key takeaway:
  The model learned to generate different-sounding names for each
  category, even from just 6 examples each! Teacher forcing made
  training stable — at every step, the model saw the correct previous
  character, so it could focus on learning character patterns.
""")


if __name__ == "__main__":
    train()

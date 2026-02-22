"""
Subtask 3: Training the Fantasy Name Generator

This script:
1. Generates the Fantasy Name Dataset (procedural, no download needed)
2. Trains the OneToManyLSTM model with scheduled sampling
3. Generates sample names for each category
4. Plots training curves

Your task: Complete the TODO sections for train_step() and
generate_names_for_all_categories().
"""

import argparse
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from subtask3_model import OneToManyLSTM


# =============================================================================
# Fantasy Name Dataset
# =============================================================================

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

CATEGORIES = ["elf", "dwarf", "dragon", "fairy"]
NUM_CATEGORIES = len(CATEGORIES)


def build_char_vocab():
    """Build character-level vocabulary with special tokens."""
    special = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    chars = list("abcdefghijklmnopqrstuvwxyz")
    all_tokens = special + chars
    char_to_idx = {ch: i for i, ch in enumerate(all_tokens)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


# Phonetic syllable rules for each category
SYLLABLE_RULES = {
    "elf": {
        "onsets": [
            "Thal", "Ser", "El", "Aer", "Gal", "Cel", "Lir", "Nal", "Syl", "Eil",
            "Fen", "Mir", "Lor", "Ar", "Il", "Val", "Ael", "Ith", "Ori", "Vel",
        ],
        "middles": [
            "in", "en", "an", "il", "ar", "al", "iel", "wen", "dri", "ori",
            "eth", "ith", "and", "ael", "ion", "ell", "ind", "aer",
        ],
        "endings": [
            "dra", "iel", "wen", "ara", "is", "eth", "a", "en", "orn", "ath",
            "ra", "na", "la", "ril", "iel", "wen",
        ],
    },
    "dwarf": {
        "onsets": [
            "Gr", "Th", "B", "Dr", "K", "Br", "G", "D", "Kr", "Tr",
            "Sk", "St", "Bl", "Gl", "Kh", "Dh", "Gn", "Sn", "Gor", "Bor",
        ],
        "middles": [
            "um", "or", "az", "un", "ag", "ur", "im", "ok", "ar", "ul",
            "on", "ak", "ug", "ek", "oz", "ud",
        ],
        "endings": [
            "nak", "din", "rik", "grim", "gor", "dur", "bak", "rok",
            "dok", "bur", "mak", "dak", "gar", "nur", "gin", "gur",
        ],
    },
    "dragon": {
        "onsets": [
            "Zyr", "Vex", "Sh", "Kra", "Thr", "Xar", "Vol", "Dra", "Vy", "Syx",
            "Mor", "Nyx", "Gor", "Bal", "Raz", "Tor", "Zar", "Kry", "Var", "Tyx",
        ],
        "middles": [
            "ax", "ith", "ag", "ex", "oth", "ix", "az", "yr",
            "aex", "eth", "ox", "yz", "ux", "ath", "ek", "or",
        ],
        "endings": [
            "oth", "yr", "ax", "ion", "us", "is", "eth", "yx",
            "or", "ax", "oth", "an", "os", "ith", "ur", "on",
        ],
    },
    "fairy": {
        "onsets": [
            "Pix", "Tw", "Fl", "Gl", "Sp", "Br", "Sh", "Bl", "Li", "Ti",
            "Sw", "Ri", "Wi", "De", "Lu", "Si", "Cr", "St", "Mi", "Fi",
        ],
        "middles": [
            "el", "in", "yl", "an", "im", "ar", "ow", "il", "ee", "ay",
            "al", "er", "en", "ol", "or", "un",
        ],
        "endings": [
            "la", "na", "a", "ie", "ette", "ine", "ia", "ee", "y",
            "elle", "ina", "ora", "ana", "isa", "ala", "ey",
        ],
    },
}


def generate_fantasy_names(num_per_category=400, seed=42):
    """
    Generate fantasy names using phonetic syllable rules.

    Each category has distinct phonetic patterns:
    - Elf: flowing, vowel-heavy (Thalindra, Sereniel, Elowen)
    - Dwarf: hard consonants, guttural (Grumnak, Thordin, Bazrik)
    - Dragon: harsh, majestic (Zyraxoth, Vexithyr, Shaedrax)
    - Fairy: light, musical (Pixella, Twylana, Fleena)

    Returns:
        List of (category_name, name_string) tuples
    """
    random.seed(seed)
    all_names = []

    for cat_name in CATEGORIES:
        rules = SYLLABLE_RULES[cat_name]
        names_set = set()

        attempts = 0
        while len(names_set) < num_per_category and attempts < num_per_category * 20:
            attempts += 1
            name = (
                random.choice(rules["onsets"])
                + random.choice(rules["middles"])
                + random.choice(rules["endings"])
            )

            # Deduplicate (case-insensitive)
            if name.lower() not in {n.lower() for n in names_set}:
                names_set.add(name)

        for name in sorted(names_set):
            all_names.append((cat_name, name))

    return all_names


class FantasyNameDataset(Dataset):
    """
    PyTorch Dataset for fantasy names.

    Each sample is (category_index, encoded_name_tensor).
    The encoded name includes SOS and EOS tokens.
    """

    def __init__(self, names, char_to_idx):
        self.data = []
        self.char_to_idx = char_to_idx
        sos_idx = char_to_idx[SOS_TOKEN]
        eos_idx = char_to_idx[EOS_TOKEN]

        cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

        for cat_name, name in names:
            cat_idx = cat_to_idx[cat_name]
            # Encode: [SOS, c1, c2, ..., cn, EOS]
            char_indices = [sos_idx]
            for ch in name.lower():
                if ch in char_to_idx:
                    char_indices.append(char_to_idx[ch])
            char_indices.append(eos_idx)

            self.data.append((cat_idx, torch.tensor(char_indices, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Collate function that pads sequences to the same length.

    Args:
        batch: List of (category_idx, sequence_tensor) tuples

    Returns:
        categories: (batch_size,)
        sequences: (batch_size, max_seq_len) padded with 0
    """
    categories = torch.tensor([b[0] for b in batch], dtype=torch.long)
    sequences = [b[1] for b in batch]
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(batch), max_len), 0, dtype=torch.long)  # PAD = 0
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return categories, padded


# =============================================================================
# Training
# =============================================================================


def train_step(
    model, categories, targets, optimizer, criterion, clip_norm=1.0, teacher_forcing_ratio=1.0
):
    """
    Perform one training step.

    Args:
        model: OneToManyLSTM model
        categories: Category indices, shape (batch_size,)
        targets: Target sequences, shape (batch_size, seq_len)
        optimizer: Optimizer
        criterion: Loss function (CrossEntropyLoss with ignore_index=0)
        clip_norm: Max gradient norm for clipping
        teacher_forcing_ratio: Probability of using ground truth

    Returns:
        loss_value: Float loss for this step
    """
    # TODO 1: Implement the training step
    # Steps:
    # 1. Zero gradients: optimizer.zero_grad()
    # 2. Forward pass: logits = model(categories, targets, teacher_forcing_ratio)
    # 3. Reshape for loss computation:
    #    - logits: (batch_size, seq_len-1, vocab_size) -> (batch_size*(seq_len-1), vocab_size)
    #    - target: targets[:, 1:] flattened to (batch_size*(seq_len-1),)
    #    (We skip targets[:, 0] because that's the SOS input, not a prediction target)
    # 4. Compute loss: loss = criterion(logits_flat, targets_flat)
    # 5. Backward: loss.backward()
    # 6. Clip gradients: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    # 7. Update weights: optimizer.step()
    # 8. Return loss.item()
    raise NotImplementedError("TODO 1: Implement train_step()")


def train_epoch(model, dataloader, optimizer, criterion, teacher_forcing_ratio=1.0, clip_norm=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for categories, targets in dataloader:
        loss = train_step(
            model, categories, targets, optimizer, criterion, clip_norm, teacher_forcing_ratio
        )
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


# =============================================================================
# Name Generation
# =============================================================================


def generate_names_for_all_categories(
    model, char_to_idx, idx_to_char, num_names=5, temperature=0.7
):
    """
    Generate sample names for every category and print them.

    Args:
        model: Trained OneToManyLSTM model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        num_names: Number of names to generate per category
        temperature: Sampling temperature
    """
    model.eval()
    sos_idx = char_to_idx[SOS_TOKEN]
    eos_idx = char_to_idx[EOS_TOKEN]

    # TODO 2: Generate names for all categories
    # Steps:
    # 1. Loop over each category (enumerate CATEGORIES):
    #    a. Initialize list for this category's names
    #    b. Generate num_names names:
    #       - Call model.generate(cat_idx, sos_idx, eos_idx, temperature=temperature)
    #       - Convert returned indices to characters using idx_to_char
    #       - Join characters into a string: ''.join(idx_to_char[i] for i in indices)
    #       - Capitalize first letter: name.capitalize()
    #       - Append to list
    #    c. Print: f"  {cat_name.capitalize():8s}: {', '.join(names)}"
    raise NotImplementedError("TODO 2: Implement generate_names_for_all_categories()")


# =============================================================================
# Visualization
# =============================================================================


def plot_training_curves(train_losses, output_dir):
    """Plot and save training loss curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, "b-", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fantasy Name Generator - Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"\nTraining curves saved to {output_dir}/training_curves.png")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Fantasy Name Generator")
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Use mini dataset (100 names/category, 15 epochs) for fast iteration",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden size (default: 128)")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature (default: 0.7)"
    )
    args = parser.parse_args()

    # Settings based on mode
    if args.mini:
        num_per_category = 100
        num_epochs = args.epochs or 15
        print_every = 3
    else:
        num_per_category = 400
        num_epochs = args.epochs or 50
        print_every = 5

    output_dir = "results/subtask3"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Fantasy Name Generator - Training")
    if args.mini:
        print("  [MINI MODE: 100 names/category, fewer epochs]")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Build vocabulary
    char_to_idx, idx_to_char = build_char_vocab()
    vocab_size = len(char_to_idx)
    pad_idx = char_to_idx[PAD_TOKEN]

    print(f"\nVocabulary size: {vocab_size}")
    print(f"Categories: {CATEGORIES}")

    # Generate dataset
    print(f"\nGenerating fantasy names ({num_per_category} per category)...")
    all_names = generate_fantasy_names(num_per_category=num_per_category)
    print(f"Total names: {len(all_names)}")

    # Show samples
    print("\nSample names:")
    for cat in CATEGORIES:
        cat_names = [n for c, n in all_names if c == cat][:5]
        print(f"  {cat.capitalize():8s}: {', '.join(cat_names)}")

    # Create dataset and dataloader
    dataset = FantasyNameDataset(all_names, char_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Create model
    model = OneToManyLSTM(
        vocab_size=vocab_size,
        num_categories=NUM_CATEGORIES,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("Teacher forcing: decaying from 1.0 to 0.1\n")

    train_losses = []

    for epoch in range(1, num_epochs + 1):
        # Decay teacher forcing ratio
        tf_ratio = max(0.1, 1.0 - (epoch - 1) / num_epochs)

        # Train
        avg_loss = train_epoch(
            model, dataloader, optimizer, criterion, teacher_forcing_ratio=tf_ratio
        )
        train_losses.append(avg_loss)

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.4f} | TF Ratio: {tf_ratio:.2f}")

            # Generate sample names
            print("  Sample generations:")
            for cat_idx, cat_name in enumerate(CATEGORIES):
                with torch.no_grad():
                    indices = model.generate(
                        cat_idx,
                        char_to_idx[SOS_TOKEN],
                        char_to_idx[EOS_TOKEN],
                        temperature=args.temperature,
                    )
                    name = "".join(idx_to_char[i] for i in indices).capitalize()
                    print(f"    {cat_name.capitalize():8s}: {name}")
            print()

    # Save model
    model_path = os.path.join(output_dir, "name_generator.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training curves
    plot_training_curves(train_losses, output_dir)

    # Final generation
    print("\n" + "=" * 60)
    print(f"Final Name Generation (temperature = {args.temperature:.1f})")
    print("=" * 60)
    generate_names_for_all_categories(
        model, char_to_idx, idx_to_char, num_names=8, temperature=args.temperature
    )

    # Also show different temperatures
    print("\n" + "=" * 60)
    print("Temperature Comparison")
    print("=" * 60)
    for temp in [0.3, 0.7, 1.0, 1.5]:
        print(f"\nTemperature = {temp}:")
        generate_names_for_all_categories(
            model, char_to_idx, idx_to_char, num_names=3, temperature=temp
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

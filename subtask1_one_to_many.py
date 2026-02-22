"""
Subtask 1: One-to-Many RNN - From Category to Sequence

The previous lab focused on Many-to-One RNNs (sentiment analysis):
  Input sequence -> Single output (positive/negative)

This lab explores the REVERSE: One-to-Many RNNs:
  Single input -> Output sequence (generated name)

Use case: Given a fantasy creature category (Elf, Dwarf, Dragon, Fairy),
generate a character-level name with phonetic patterns distinct to each race.

Your task: Complete the TODO sections to implement the one-to-many RNN cell.
"""

import torch
import torch.nn as nn


# =============================================================================
# PART 1: Many-to-One vs One-to-Many
# =============================================================================


def explain_architectures():
    """Compare many-to-one and one-to-many architectures."""
    print("=" * 60)
    print("PART 1: Many-to-One vs One-to-Many")
    print("=" * 60)

    print("""
MANY-TO-ONE (Previous Lab - Sentiment Analysis):

    Input:  x1    x2    x3    x4    x5
             |     |     |     |     |
             v     v     v     v     v
    RNN:   [h1]--[h2]--[h3]--[h4]--[h5]
                                     |
                                     v
    Output:                       [class]

    "The movie was really great" -> Positive

    The ENTIRE sequence is consumed to produce ONE output.


ONE-TO-MANY (This Lab - Name Generation):

    Input:  [category]
                |
                v
    RNN:      [h0]--[h1]--[h2]--[h3]--[h4]--[h5]
               |     |     |     |     |     |
               v     v     v     v     v     v
    Output:  SOS    'T'   'h'   'a'   'l'  EOS

    "Elf" -> "Thal"

    ONE input produces a SEQUENCE of outputs (characters).

    Key differences:
    - Many-to-one: Multiple inputs, one output at the end
    - One-to-many: One input, multiple outputs over time
    - The hidden state carries information in BOTH directions
    """)


# =============================================================================
# PART 2: Special Tokens and Character Vocabulary
# =============================================================================

# Special tokens for sequence generation
PAD_TOKEN = "<PAD>"  # Padding for batching
SOS_TOKEN = "<SOS>"  # Start Of Sequence - signals "start generating"
EOS_TOKEN = "<EOS>"  # End Of Sequence - signals "stop generating"


def build_char_vocab():
    """
    Build a character-level vocabulary for name generation.

    Returns:
        char_to_idx: Dict mapping characters to indices
        idx_to_char: Dict mapping indices back to characters
    """
    # Special tokens get the first indices
    special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]

    # All lowercase letters
    chars = list("abcdefghijklmnopqrstuvwxyz")

    all_tokens = special_tokens + chars

    char_to_idx = {ch: i for i, ch in enumerate(all_tokens)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    return char_to_idx, idx_to_char


def explain_special_tokens():
    """Explain why we need special tokens for generation."""
    print("\n" + "=" * 60)
    print("PART 2: Special Tokens for Sequence Generation")
    print("=" * 60)

    char_to_idx, idx_to_char = build_char_vocab()

    print(f"""
When GENERATING sequences, we need special tokens:

  <PAD> (index {char_to_idx[PAD_TOKEN]}): Padding for batching variable-length sequences
  <SOS> (index {char_to_idx[SOS_TOKEN]}): Start Of Sequence - the first input to the decoder
  <EOS> (index {char_to_idx[EOS_TOKEN]}): End Of Sequence - tells the model to stop

Example encoding of "Thal":
  Input to decoder:  <SOS>  ->  t  ->  h  ->  a  ->  l
  Target output:       t   ->  h  ->  a  ->  l  -> <EOS>

The model learns to:
  1. Start generating when it sees <SOS>
  2. Output characters one at a time
  3. Output <EOS> when the name is complete

Vocabulary size: {len(char_to_idx)} (3 special + {len(char_to_idx) - 3} letters)
""")

    return char_to_idx, idx_to_char


# =============================================================================
# PART 3: One-to-Many RNN Cell
# =============================================================================

# Category definitions
CATEGORIES = ["elf", "dwarf", "dragon", "fairy"]
NUM_CATEGORIES = len(CATEGORIES)


class OneToManyRNNCell(nn.Module):
    """
    A one-to-many RNN that generates character sequences from a category.

    Architecture:
    1. ENCODE: Category -> Hidden state (one-to-many "one" part)
    2. DECODE: Hidden state -> Characters one at a time (one-to-many "many" part)

    Components:
    - category_embedding: Maps category index to a dense vector
    - category_to_hidden: Projects category embedding to initial hidden state
    - category_to_cell: Projects category embedding to initial cell state (LSTM)
    - char_embedding: Maps character index to a dense vector
    - lstm_cell: LSTMCell that processes one character at a time
    - output_projection: Maps hidden state to character probabilities
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_size: int = 64,
        num_categories: int = NUM_CATEGORIES,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Encoder: category -> hidden state
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.category_to_hidden = nn.Linear(embed_dim, hidden_size)
        self.category_to_cell = nn.Linear(embed_dim, hidden_size)

        # Decoder: character-level LSTM
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def encode(self, category_idx: torch.Tensor):
        """
        Encode a category into an initial hidden state.

        This is the "one" part of one-to-many: a single category input
        gets transformed into the initial hidden state that will drive
        the entire sequence generation.

        Args:
            category_idx: Category index tensor, shape (batch_size,)

        Returns:
            h0: Initial hidden state, shape (batch_size, hidden_size)
            c0: Initial cell state, shape (batch_size, hidden_size)
        """
        # TODO 1: Encode the category into initial hidden and cell states
        # Steps:
        # 1. Pass category_idx through self.category_embedding to get category vector
        # 2. Pass the category vector through self.category_to_hidden and apply tanh
        #    to get h0
        # 3. Pass the category vector through self.category_to_cell and apply tanh
        #    to get c0
        # 4. Return (h0, c0)
        raise NotImplementedError("TODO 1: Implement OneToManyRNNCell.encode()")

    def decode_step(self, char_idx: torch.Tensor, hidden: tuple):
        """
        Perform ONE step of decoding: given a character and hidden state,
        produce the next character's logits and updated hidden state.

        Args:
            char_idx: Current character index, shape (batch_size,)
            hidden: Tuple of (h, c) each shape (batch_size, hidden_size)

        Returns:
            logits: Predicted next character logits, shape (batch_size, vocab_size)
            hidden: Updated (h, c) tuple
        """
        # TODO 2: Implement one decoding step
        # Steps:
        # 1. Embed the character: pass char_idx through self.char_embedding
        # 2. Run one LSTM step: new_hidden = self.lstm_cell(embedded, hidden)
        # 3. Project hidden state to vocabulary: logits = self.output_projection(h)
        #    where h is the hidden state (first element) from new_hidden
        # 4. Return (logits, new_hidden)
        raise NotImplementedError("TODO 2: Implement OneToManyRNNCell.decode_step()")


# =============================================================================
# PART 4: Greedy Generation
# =============================================================================


def generate_greedy(
    model: OneToManyRNNCell,
    category_idx: int,
    char_to_idx: dict,
    idx_to_char: dict,
    max_len: int = 20,
) -> str:
    """
    Generate a name using greedy decoding (always pick the most likely character).

    The generation loop:
    1. Encode the category to get initial hidden state
    2. Start with <SOS> token
    3. At each step, pick the character with highest probability
    4. Feed that character back as input to the next step
    5. Stop when <EOS> is generated or max_len is reached

    Args:
        model: OneToManyRNNCell model
        category_idx: Integer category index
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        max_len: Maximum name length

    Returns:
        generated_name: The generated name string
    """
    model.eval()

    sos_idx = char_to_idx[SOS_TOKEN]
    eos_idx = char_to_idx[EOS_TOKEN]

    # TODO 3: Implement greedy generation
    # Steps:
    # 1. Create category tensor: torch.tensor([category_idx]) (batch of 1)
    # 2. Encode: hidden = model.encode(category_tensor)
    # 3. Start with SOS token: current_char = torch.tensor([sos_idx])
    # 4. Initialize empty list for generated characters
    # 5. Use torch.no_grad() context and loop up to max_len times:
    #    a. Get logits and new hidden: logits, hidden = model.decode_step(current_char, hidden)
    #    b. Pick the character with highest logit: next_idx = logits.argmax(dim=-1)
    #    c. If next_idx.item() == eos_idx, break
    #    d. Convert to character and append to list: idx_to_char[next_idx.item()]
    #    e. Set current_char = next_idx for the next step
    # 6. Join characters into a string and return it
    raise NotImplementedError("TODO 3: Implement generate_greedy()")


# =============================================================================
# PART 5: Demo - Untrained Model Generates Gibberish
# =============================================================================


def demo_untrained_model():
    """Show that an untrained model generates random gibberish."""
    print("\n" + "=" * 60)
    print("PART 5: Untrained Model Demo")
    print("=" * 60)

    char_to_idx, idx_to_char = build_char_vocab()
    vocab_size = len(char_to_idx)

    # Create untrained model
    torch.manual_seed(42)
    model = OneToManyRNNCell(vocab_size)

    print("\nGenerating names with an UNTRAINED model:")
    print("(These will be gibberish - the model hasn't learned anything yet!)\n")

    for cat_idx, cat_name in enumerate(CATEGORIES):
        names = []
        for _ in range(3):
            name = generate_greedy(model, cat_idx, char_to_idx, idx_to_char)
            names.append(name if name else "(empty)")
        print(f"  {cat_name.capitalize():8s}: {', '.join(names)}")

    print("""
As expected, the untrained model generates gibberish!
After training (subtask 3), it will learn:
  - Elves get flowing, vowel-heavy names (Thalindra, Sereniel)
  - Dwarves get hard, guttural names (Grumnak, Thordin)
  - Dragons get harsh, majestic names (Zyraxoth, Vexithyr)
  - Fairies get light, musical names (Pixella, Twylana)
""")


def main():
    print("\n" + "=" * 60)
    print("SUBTASK 1: One-to-Many RNN Architecture")
    print("=" * 60)

    # Part 1: Architecture comparison
    explain_architectures()

    # Part 2: Special tokens
    char_to_idx, idx_to_char = explain_special_tokens()

    # Part 3: One-to-many cell (explain)
    print("\n" + "=" * 60)
    print("PART 3: OneToManyRNNCell")
    print("=" * 60)

    vocab_size = len(char_to_idx)
    model = OneToManyRNNCell(vocab_size)

    print(f"""
The OneToManyRNNCell has two phases:

  1. ENCODE: Category -> (h0, c0)
     - Embeds category index into a dense vector
     - Projects to initial hidden and cell states

  2. DECODE: (h, c) + char -> next_char + (h', c')
     - Embeds current character
     - Runs one LSTMCell step
     - Projects hidden state to character probabilities

Model parameters: {sum(p.numel() for p in model.parameters()):,}
""")

    # Part 4: Greedy generation (explain)
    print("=" * 60)
    print("PART 4: Greedy Generation")
    print("=" * 60)
    print("""
Greedy generation picks the most likely character at each step:

  category -> encode -> h0
                        |
                        v
  <SOS>  -> decode -> h1 -> argmax -> 'T'
                                       |
                                       v
           'T' -> decode -> h2 -> argmax -> 'h'
                                             |
                                             v
           'h' -> decode -> h3 -> argmax -> 'a'
                                             |
                                             v
           'a' -> decode -> h4 -> argmax -> 'l'
                                             |
                                             v
           'l' -> decode -> h5 -> argmax -> <EOS>

  Result: "Thal"
""")

    # Part 5: Demo with untrained model
    demo_untrained_model()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    1. One-to-Many RNNs generate SEQUENCES from a SINGLE input

    2. Special tokens (<SOS>, <EOS>, <PAD>) manage sequence boundaries

    3. The architecture has two phases:
       - ENCODE: Input -> initial hidden state
       - DECODE: Hidden state -> characters, one at a time

    4. Greedy generation always picks the most likely next character

    Next: In subtask 2, we'll learn TEACHER FORCING to train this model!
""")


if __name__ == "__main__":
    main()

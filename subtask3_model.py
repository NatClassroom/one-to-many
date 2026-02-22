"""
Subtask 3: One-to-Many LSTM Model

This file defines the full OneToManyLSTM model for fantasy name generation.
It combines all concepts from subtasks 1 and 2:
  - Category encoding -> initial hidden state
  - Character-level LSTM decoding
  - Teacher forcing for training
  - Temperature sampling for generation

Your task: Complete the TODO sections for forward() and generate().
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneToManyLSTM(nn.Module):
    """
    One-to-Many LSTM model for character-level name generation.

    Given a category (Elf, Dwarf, Dragon, Fairy), generates a name
    character by character.

    Architecture:
    - Category embedding + projection -> initial (h0, c0)
    - Character embedding -> LSTMCell -> output projection
    """

    def __init__(
        self,
        vocab_size: int,
        num_categories: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Encoder: category -> hidden state
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.category_to_hidden = nn.Linear(embed_dim, hidden_size)
        self.category_to_cell = nn.Linear(embed_dim, hidden_size)

        # Decoder: character-level LSTM
        self.char_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def encode(self, categories: torch.Tensor):
        """
        Encode category indices into initial hidden states.

        Args:
            categories: shape (batch_size,)

        Returns:
            (h0, c0): each shape (batch_size, hidden_size)
        """
        cat_emb = self.category_embedding(categories)
        h0 = torch.tanh(self.category_to_hidden(cat_emb))
        c0 = torch.tanh(self.category_to_cell(cat_emb))
        return h0, c0

    def forward(
        self,
        categories: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            categories: Category indices, shape (batch_size,)
            targets: Target sequences [SOS, c1, ..., cn, EOS, PAD...],
                     shape (batch_size, seq_len)
            teacher_forcing_ratio: Probability of using ground truth at each step

        Returns:
            logits: Predicted character logits, shape (batch_size, seq_len-1, vocab_size)
                    (seq_len-1 because we don't predict from the last token)
        """
        # TODO 1: Implement the training forward pass
        # Steps:
        # 1. Encode categories: h, c = self.encode(categories)
        # 2. Extract input_seq = targets[:, :-1] (SOS + all chars, exclude last)
        #    Get num_steps = input_seq.shape[1]
        # 3. Start with SOS: current_input = input_seq[:, 0]
        # 4. Initialize list for collecting logits at each step
        # 5. Loop for each timestep t in range(num_steps):
        #    a. Embed current input: emb = self.char_embedding(current_input)
        #    b. LSTM step: (h, c) = self.lstm_cell(emb, (h, c))
        #    c. Apply dropout: h_drop = self.dropout(h)
        #    d. Project to vocabulary: step_logits = self.output_projection(h_drop)
        #    e. Append step_logits to list
        #    f. Decide next input (scheduled sampling):
        #       - If t + 1 < num_steps and random.random() < teacher_forcing_ratio:
        #           current_input = input_seq[:, t + 1]  (ground truth)
        #       - Else:
        #           current_input = step_logits.argmax(dim=-1)  (model's prediction)
        # 6. Stack all logits along dim=1: shape (batch_size, num_steps, vocab_size)
        # 7. Return the stacked logits
        raise NotImplementedError("TODO 1: Implement OneToManyLSTM.forward()")

    def generate(
        self,
        category_idx: int,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 20,
        temperature: float = 1.0,
    ) -> list:
        """
        Generate a name for the given category.

        Args:
            category_idx: Integer category index
            sos_idx: Index of SOS token
            eos_idx: Index of EOS token
            max_len: Maximum name length
            temperature: Sampling temperature (lower = more conservative)

        Returns:
            indices: List of generated character indices (excluding SOS/EOS)
        """
        # TODO 2: Implement name generation with temperature sampling
        # Steps:
        # 1. Create category tensor: torch.tensor([category_idx])
        # 2. Encode: (h, c) = self.encode(cat_tensor)
        # 3. Start with SOS: current = torch.tensor([sos_idx])
        # 4. Initialize empty list for generated indices
        # 5. Use torch.no_grad() context and loop up to max_len times:
        #    a. Embed: emb = self.char_embedding(current)
        #    b. LSTM step: (h, c) = self.lstm_cell(emb, (h, c))
        #    c. Project: logits = self.output_projection(h)
        #    d. Sample with temperature:
        #       - scaled = logits / temperature
        #       - probs = F.softmax(scaled, dim=-1)
        #       - current = torch.multinomial(probs, 1).squeeze(1)
        #    e. If current.item() == eos_idx, break
        #    f. Append current.item() to indices list
        # 6. Return the list of indices
        raise NotImplementedError("TODO 2: Implement OneToManyLSTM.generate()")


def test_model():
    """Verify model shapes and parameter count."""
    print("Testing OneToManyLSTM model...")

    vocab_size = 29  # 3 special + 26 letters
    num_categories = 4
    batch_size = 8
    seq_len = 12  # including SOS and EOS

    model = OneToManyLSTM(
        vocab_size=vocab_size,
        num_categories=num_categories,
        embed_dim=64,
        hidden_size=128,
    )

    # Test encode
    categories = torch.randint(0, num_categories, (batch_size,))
    h0, c0 = model.encode(categories)
    print(f"  Encode: categories {categories.shape} -> h0 {h0.shape}, c0 {c0.shape}")
    assert h0.shape == (batch_size, 128), f"Expected ({batch_size}, 128), got {h0.shape}"
    assert c0.shape == (batch_size, 128), f"Expected ({batch_size}, 128), got {c0.shape}"
    print("  Encode shapes correct")

    # Test forward
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(categories, targets)
    print(f"  Forward: targets {targets.shape} -> logits {logits.shape}")
    expected = (batch_size, seq_len - 1, vocab_size)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    print("  Forward shapes correct")

    # Test generate
    indices = model.generate(category_idx=0, sos_idx=1, eos_idx=2)
    print(f"  Generate: category 0 -> {len(indices)} characters")
    print("  Generate works")

    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {num_params:,}")

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_model()

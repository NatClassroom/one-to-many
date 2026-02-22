"""
Subtask 2: Teacher Forcing and Training Strategies

Training a one-to-many RNN is tricky: at each step, the model predicts
a character, and that prediction feeds into the next step. If an early
prediction is wrong, all subsequent predictions are affected — errors compound!

This exercise explores three training strategies:
  1. Teacher Forcing: Feed GROUND TRUTH characters at each step
  2. Free Running: Feed model's OWN predictions at each step
  3. Scheduled Sampling: Gradually transition from teacher forcing to free running

Plus: Temperature sampling for controlling generation diversity.

Your task: Complete the TODO sections to implement each training strategy.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subtask1_one_to_many import (
    CATEGORIES,
    EOS_TOKEN,
    NUM_CATEGORIES,
    PAD_TOKEN,
    SOS_TOKEN,
    OneToManyRNNCell,
    build_char_vocab,
)


# =============================================================================
# PART 1: The Problem - Error Compounding
# =============================================================================


def explain_error_compounding():
    print("=" * 60)
    print("PART 1: The Problem - Error Compounding")
    print("=" * 60)
    print("""
During generation, each step's output becomes the NEXT step's input.
If the model makes an error early on, it cascades:

  Target:    T -> h -> a -> l -> i -> n -> d -> r -> a -> <EOS>

  Step 1: Predict 'T' (correct!)
  Step 2: Input 'T', predict 'h' (correct!)
  Step 3: Input 'h', predict 'z' (wrong! should be 'a')
  Step 4: Input 'z', predict '?' (never saw 'z' after 'h' in training!)
  Step 5: Complete garbage from here on...

This is called EXPOSURE BIAS: during training with teacher forcing,
the model always sees correct inputs. At test time, it sees its own
(possibly wrong) outputs — a situation it never trained for!
""")


# =============================================================================
# Helper: Fantasy name mini-dataset for this exercise
# =============================================================================


def generate_mini_dataset(char_to_idx, num_per_category=50):
    """Generate a small fantasy name dataset for training strategy comparison."""
    random.seed(42)

    syllable_rules = {
        "elf": {
            "onsets": ["Th", "S", "El", "Aer", "G", "C", "L", "N", "F", "M", "Ar", "V"],
            "middles": ["al", "in", "en", "an", "il", "ar", "iel", "wen", "ori", "eth"],
            "endings": ["dra", "iel", "wen", "ara", "is", "eth", "a", "en", "orn"],
        },
        "dwarf": {
            "onsets": ["Gr", "Th", "B", "Dr", "K", "Br", "G", "D", "Kr", "Tr"],
            "middles": ["um", "or", "az", "un", "ag", "ur", "im", "ok", "ar", "ul"],
            "endings": ["nak", "din", "rik", "grim", "gor", "dur", "bak", "rok"],
        },
        "dragon": {
            "onsets": ["Zyr", "Vex", "Sh", "Kra", "Thr", "Xar", "Vol", "Dra", "Vy", "Syx"],
            "middles": ["ax", "ith", "ag", "ex", "oth", "ix", "az", "yr"],
            "endings": ["oth", "yr", "ax", "ion", "us", "is", "eth", "yx"],
        },
        "fairy": {
            "onsets": ["Pix", "Tw", "Fl", "Gl", "Sp", "Br", "Sh", "Bl", "Li", "Ti"],
            "middles": ["el", "in", "yl", "an", "im", "ar", "ow", "il", "ee"],
            "endings": ["la", "na", "a", "ie", "ette", "ine", "ia", "ee", "y"],
        },
    }

    sos_idx = char_to_idx[SOS_TOKEN]
    eos_idx = char_to_idx[EOS_TOKEN]
    pad_idx = char_to_idx[PAD_TOKEN]

    data = []  # List of (category_idx, target_indices)

    for cat_idx, cat_name in enumerate(CATEGORIES):
        rules = syllable_rules[cat_name]
        names_set = set()

        while len(names_set) < num_per_category:
            name = (
                random.choice(rules["onsets"])
                + random.choice(rules["middles"])
                + random.choice(rules["endings"])
            )
            name_lower = name.lower()
            if name_lower not in names_set:
                names_set.add(name_lower)
                # Convert to indices: [SOS, c1, c2, ..., EOS]
                indices = (
                    [sos_idx]
                    + [char_to_idx.get(c, pad_idx) for c in name_lower]
                    + [eos_idx]
                )
                data.append((cat_idx, torch.tensor(indices, dtype=torch.long)))

    random.shuffle(data)
    return data


def collate_names(batch, pad_idx=0):
    """Pad a batch of (category, target_sequence) to the same length."""
    categories = torch.tensor([b[0] for b in batch], dtype=torch.long)
    sequences = [b[1] for b in batch]
    max_len = max(len(s) for s in sequences)
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return categories, padded


# =============================================================================
# PART 2: Teacher Forcing
# =============================================================================


def explain_teacher_forcing():
    print("\n" + "=" * 60)
    print("PART 2: Teacher Forcing")
    print("=" * 60)
    print("""
TEACHER FORCING feeds the GROUND TRUTH character at each step,
regardless of what the model predicted:

  Target:    <SOS> -> T -> h -> a -> l -> <EOS>

  Step 1: Input <SOS>, predict logits_1 -> loss vs 'T'
  Step 2: Input 'T'  , predict logits_2 -> loss vs 'h'   <- ground truth 'T'
  Step 3: Input 'h'  , predict logits_3 -> loss vs 'a'   <- ground truth 'h'
  Step 4: Input 'a'  , predict logits_4 -> loss vs 'l'
  Step 5: Input 'l'  , predict logits_5 -> loss vs <EOS>

Advantages:
  + Fast, stable training (no error compounding during training)
  + Model always sees correct context

Disadvantages:
  - Exposure bias: model never learns to recover from its own mistakes
  - At inference time, there is no "teacher" — model uses its own outputs
""")


def compute_loss_with_teacher_forcing(model, categories, targets, pad_idx, criterion):
    """
    Compute loss using teacher forcing: feed ground truth at each step.

    Args:
        model: OneToManyRNNCell
        categories: Category indices, shape (batch_size,)
        targets: Target sequences including SOS and EOS, shape (batch_size, seq_len)
                 Format: [SOS, c1, c2, ..., cn, EOS, PAD, PAD, ...]
        pad_idx: Index of PAD token
        criterion: Loss function (CrossEntropyLoss with ignore_index=pad_idx)

    Returns:
        loss: Scalar loss value
    """
    # TODO 1: Implement teacher-forced training
    # Steps:
    # 1. Encode categories to get initial hidden state: hidden = model.encode(categories)
    # 2. The input sequence is targets[:, :-1] (everything except last token)
    #    The expected output is targets[:, 1:] (everything except first token, i.e. SOS)
    # 3. Initialize a list to collect logits at each step
    # 4. Loop over each timestep t in range(input_seq.shape[1]):
    #    a. Get the input character at step t: input_char = input_seq[:, t]
    #    b. Run one decode step: logits, hidden = model.decode_step(input_char, hidden)
    #    c. Append logits to the list
    # 5. Stack all logits: torch.stack(all_logits, dim=1) -> shape (batch_size, num_steps, vocab_size)
    # 6. Reshape for loss: logits as (batch_size * num_steps, vocab_size)
    #    and target_seq as (batch_size * num_steps,)
    # 7. Compute and return: criterion(logits_flat, targets_flat)
    raise NotImplementedError("TODO 1: Implement compute_loss_with_teacher_forcing()")


# =============================================================================
# PART 3: Free Running
# =============================================================================


def explain_free_running():
    print("\n" + "=" * 60)
    print("PART 3: Free Running (Autoregressive)")
    print("=" * 60)
    print("""
FREE RUNNING feeds the model's OWN prediction at each step:

  Target:    <SOS> -> T -> h -> a -> l -> <EOS>

  Step 1: Input <SOS>, predict logits_1 -> argmax -> 'T' -> loss vs 'T'
  Step 2: Input 'T'  , predict logits_2 -> argmax -> 'h' -> loss vs 'h'
  Step 3: Input 'h'  , predict logits_3 -> argmax -> 'z' -> loss vs 'a'  <- used 'z'!
  Step 4: Input 'z'  , predict logits_4 -> argmax -> '?' -> loss vs 'l'
  Step 5: Input '?'  , predict logits_5 -> argmax -> '?' -> loss vs <EOS>

Advantages:
  + No exposure bias -- model practices with its own outputs
  + What you train is what you get at inference

Disadvantages:
  - Very unstable early in training (predictions are random)
  - Slower convergence
""")


def compute_loss_free_running(model, categories, targets, pad_idx, criterion):
    """
    Compute loss using free running: feed model's own predictions.

    Args:
        model: OneToManyRNNCell
        categories: Category indices, shape (batch_size,)
        targets: Target sequences, shape (batch_size, seq_len)
        pad_idx: Index of PAD token
        criterion: Loss function

    Returns:
        loss: Scalar loss value
    """
    # TODO 2: Implement free-running training
    # Steps:
    # 1. Encode: hidden = model.encode(categories)
    # 2. Extract target_seq = targets[:, 1:] (what we want the model to produce)
    #    num_steps = target_seq.shape[1]
    # 3. Start with SOS: current_input = targets[:, 0] (the SOS token)
    # 4. Initialize list for logits
    # 5. Loop for num_steps:
    #    a. Decode one step: logits, hidden = model.decode_step(current_input, hidden)
    #    b. Append logits
    #    c. Pick the model's own prediction: current_input = logits.argmax(dim=-1)
    #       (This is the key difference from teacher forcing!)
    # 6. Stack logits: torch.stack(all_logits, dim=1) -> (batch_size, num_steps, vocab_size)
    # 7. Reshape and compute loss (same as teacher forcing steps 6-7)
    raise NotImplementedError("TODO 2: Implement compute_loss_free_running()")


# =============================================================================
# PART 4: Scheduled Sampling
# =============================================================================


def explain_scheduled_sampling():
    print("\n" + "=" * 60)
    print("PART 4: Scheduled Sampling")
    print("=" * 60)
    print("""
SCHEDULED SAMPLING mixes teacher forcing and free running:

At each step, flip a coin (weighted by teacher_forcing_ratio):
  - With probability p: use ground truth (teacher forcing)
  - With probability 1-p: use model's prediction (free running)

  p starts at 1.0 (100% teacher forcing) and decays over training.

  Early training: p ~ 1.0 -> mostly teacher forcing (stable learning)
  Late training:  p ~ 0.0 -> mostly free running (learns to self-correct)

This is the BEST OF BOTH WORLDS:
  + Stable early training (like teacher forcing)
  + No exposure bias at convergence (like free running)
""")


def compute_loss_scheduled_sampling(
    model, categories, targets, pad_idx, criterion, teacher_forcing_ratio=0.5
):
    """
    Compute loss using scheduled sampling: probabilistic mix of teacher forcing
    and free running at each timestep.

    Args:
        model: OneToManyRNNCell
        categories: Category indices, shape (batch_size,)
        targets: Target sequences, shape (batch_size, seq_len)
        pad_idx: Index of PAD token
        criterion: Loss function
        teacher_forcing_ratio: Probability of using ground truth at each step

    Returns:
        loss: Scalar loss value
    """
    # TODO 3: Implement scheduled sampling
    # Steps:
    # 1. Encode: hidden = model.encode(categories)
    # 2. Extract input_seq = targets[:, :-1] and target_seq = targets[:, 1:]
    #    num_steps = target_seq.shape[1]
    # 3. Start with SOS: current_input = input_seq[:, 0]
    # 4. Initialize list for logits
    # 5. Loop over each timestep t in range(num_steps):
    #    a. Decode: logits, hidden = model.decode_step(current_input, hidden)
    #    b. Append logits
    #    c. Decide what to feed next (if t + 1 < num_steps):
    #       - If random.random() < teacher_forcing_ratio:
    #           current_input = input_seq[:, t + 1]
    #         (use ground truth from input sequence)
    #       - Else:
    #           current_input = logits.argmax(dim=-1)
    #         (use model's own prediction)
    # 6. Stack logits: torch.stack(all_logits, dim=1)
    # 7. Reshape and compute loss (same as before)
    #
    # Note: Step 5c is where the "scheduled" part happens -- we randomly
    # choose between teacher forcing and free running at each step.
    raise NotImplementedError("TODO 3: Implement compute_loss_scheduled_sampling()")


# =============================================================================
# PART 5: Temperature Sampling
# =============================================================================


def explain_temperature():
    print("\n" + "=" * 60)
    print("PART 5: Temperature Sampling")
    print("=" * 60)
    print("""
TEMPERATURE controls the "randomness" of generation:

  softmax(logits / temperature)

  Temperature = 1.0: Normal probabilities (default)
  Temperature < 1.0: Sharper (more confident, less diverse)
  Temperature > 1.0: Flatter (less confident, more diverse)
  Temperature -> 0.0: Greedy (always pick most likely)
  Temperature -> inf : Uniform random

Example with logits [2.0, 1.0, 0.5]:
  T=0.5: softmax([4.0, 2.0, 1.0]) = [0.84, 0.11, 0.04]  <- very confident
  T=1.0: softmax([2.0, 1.0, 0.5]) = [0.56, 0.21, 0.13]  <- moderate
  T=2.0: softmax([1.0, 0.5, 0.25]) = [0.39, 0.24, 0.18]  <- more uniform
""")


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from logits with temperature scaling.

    Args:
        logits: Raw logits, shape (batch_size, vocab_size)
        temperature: Temperature for scaling (default 1.0)

    Returns:
        sampled_indices: Sampled character indices, shape (batch_size,)
    """
    # TODO 4: Implement temperature sampling
    # Steps:
    # 1. Divide logits by temperature: scaled = logits / temperature
    # 2. Convert to probabilities: probs = F.softmax(scaled, dim=-1)
    # 3. Sample from the distribution: torch.multinomial(probs, num_samples=1)
    # 4. Squeeze out the last dimension and return: shape should be (batch_size,)
    raise NotImplementedError("TODO 4: Implement sample_with_temperature()")


# =============================================================================
# PART 6: Training Comparison
# =============================================================================


def train_with_strategy(
    strategy_name, loss_fn, model, data, char_to_idx, num_epochs=30, lr=0.003, batch_size=32
):
    """Train with a given strategy and return losses."""
    pad_idx = char_to_idx[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        random.shuffle(data)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            if len(batch) < 2:
                continue

            categories, targets = collate_names(batch, pad_idx)

            optimizer.zero_grad()

            if strategy_name == "scheduled_sampling":
                # Decay teacher forcing ratio over epochs
                tf_ratio = max(0.1, 1.0 - epoch / num_epochs)
                loss = loss_fn(
                    model,
                    categories,
                    targets,
                    pad_idx,
                    criterion,
                    teacher_forcing_ratio=tf_ratio,
                )
            else:
                loss = loss_fn(model, categories, targets, pad_idx, criterion)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}")

    return losses


def compare_strategies():
    """Compare all three training strategies."""
    print("\n" + "=" * 60)
    print("PART 6: Training Strategy Comparison")
    print("=" * 60)

    char_to_idx, idx_to_char = build_char_vocab()
    vocab_size = len(char_to_idx)
    data = generate_mini_dataset(char_to_idx, num_per_category=50)

    strategies = {
        "teacher_forcing": compute_loss_with_teacher_forcing,
        "free_running": compute_loss_free_running,
        "scheduled_sampling": compute_loss_scheduled_sampling,
    }

    all_losses = {}
    num_epochs = 30

    for name, loss_fn in strategies.items():
        print(f"\nTraining with {name}...")
        torch.manual_seed(42)
        model = OneToManyRNNCell(vocab_size)
        losses = train_with_strategy(
            name, loss_fn, model, data, char_to_idx, num_epochs=num_epochs
        )
        all_losses[name] = losses

    # Plot comparison
    plt.figure(figsize=(10, 5))
    for name, losses in all_losses.items():
        plt.plot(
            range(1, num_epochs + 1),
            losses,
            label=name.replace("_", " ").title(),
            marker="o",
            markersize=3,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Strategy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results/subtask2", exist_ok=True)
    plt.savefig("results/subtask2/strategy_comparison.png", dpi=150)
    plt.close()
    print("\nPlot saved to results/subtask2/strategy_comparison.png")

    print("""
Observations:
  - Teacher forcing: Lowest loss (it's the easiest training signal)
  - Free running: Highest loss (especially early on -- error compounding!)
  - Scheduled sampling: Middle ground -- starts like TF, converges well

  In practice, scheduled sampling often gives the best GENERATION quality
  because the model learns to recover from its own mistakes.
""")


def demo_temperature():
    """Demonstrate temperature sampling with a trained model."""
    print("\n" + "=" * 60)
    print("Temperature Sampling Demo")
    print("=" * 60)

    char_to_idx, idx_to_char = build_char_vocab()
    vocab_size = len(char_to_idx)
    sos_idx = char_to_idx[SOS_TOKEN]
    eos_idx = char_to_idx[EOS_TOKEN]

    # Quick-train a small model
    torch.manual_seed(42)
    model = OneToManyRNNCell(vocab_size)
    data = generate_mini_dataset(char_to_idx, num_per_category=50)
    pad_idx = char_to_idx[PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("Quick-training model (20 epochs)...")
    for epoch in range(20):
        random.shuffle(data)
        for i in range(0, len(data), 32):
            batch = data[i : i + 32]
            if len(batch) < 2:
                continue
            categories, targets = collate_names(batch, pad_idx)
            optimizer.zero_grad()
            loss = compute_loss_with_teacher_forcing(
                model, categories, targets, pad_idx, criterion
            )
            loss.backward()
            optimizer.step()

    print("\nGenerating with different temperatures:\n")
    model.eval()

    for temp in [0.3, 0.7, 1.0, 1.5]:
        print(f"  Temperature = {temp}:")
        for cat_idx, cat_name in enumerate(CATEGORIES):
            names = []
            for _ in range(3):
                with torch.no_grad():
                    cat_tensor = torch.tensor([cat_idx])
                    hidden = model.encode(cat_tensor)
                    current = torch.tensor([sos_idx])
                    chars = []
                    for _ in range(20):
                        logits, hidden = model.decode_step(current, hidden)
                        current = sample_with_temperature(logits, temp)
                        if current.item() == eos_idx:
                            break
                        ch = idx_to_char.get(current.item(), "?")
                        if ch not in ("<PAD>", "<SOS>", "<EOS>"):
                            chars.append(ch)
                    names.append("".join(chars) if chars else "(empty)")
            print(f"    {cat_name.capitalize():8s}: {', '.join(names)}")
        print()

    print("""
  Low temperature (0.3): More repetitive, "safe" names
  Medium temperature (0.7): Good balance of quality and diversity
  High temperature (1.5): More creative but sometimes nonsensical
""")


def main():
    print("\n" + "=" * 60)
    print("SUBTASK 2: Teacher Forcing and Training Strategies")
    print("=" * 60)

    # Part 1: The problem
    explain_error_compounding()

    # Part 2: Teacher forcing
    explain_teacher_forcing()

    # Part 3: Free running
    explain_free_running()

    # Part 4: Scheduled sampling
    explain_scheduled_sampling()

    # Part 5: Temperature
    explain_temperature()

    # Part 6: Compare strategies
    compare_strategies()

    # Demo temperature
    demo_temperature()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    1. TEACHER FORCING: Fast training, but exposure bias at test time

    2. FREE RUNNING: No bias, but unstable training

    3. SCHEDULED SAMPLING: Best of both -- gradually transitions from
       teacher forcing to free running over the course of training

    4. TEMPERATURE: Controls generation diversity
       - Low: Confident, repetitive
       - High: Creative, potentially nonsensical

    Next: In subtask 3, we'll build and train the full model!
""")


if __name__ == "__main__":
    main()

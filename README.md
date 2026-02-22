# One-to-Many RNN Lab: Fantasy Name Generator

## Overview

This lab teaches **one-to-many RNN** architectures — models that take a single input and produce a sequence output. Building on the previous lab (many-to-one for sentiment analysis), you'll implement a **Fantasy Name Generator** that creates character-level names for different creature categories (Elf, Dwarf, Dragon, Fairy).

No external datasets needed — all names are procedurally generated from phonetic syllable rules.

## Setup

```bash
uv sync
```

## Subtasks

### Subtask 1: One-to-Many Architecture (`subtask1_one_to_many.py`)

Learn the one-to-many RNN architecture by implementing the core building blocks:

- **TODO 1**: `encode()` — Transform a category into an initial hidden state (~5 lines)
- **TODO 2**: `decode_step()` — Run one LSTMCell step to predict the next character (~5 lines)
- **TODO 3**: `generate_greedy()` — Full greedy generation loop (~15 lines)

```bash
uv run python subtask1_one_to_many.py
```

### Subtask 2: Teacher Forcing & Training Strategies (`subtask2_teacher_forcing.py`)

Explore how to train sequence generators:

- **TODO 1**: `compute_loss_with_teacher_forcing()` — Feed ground truth at each step (~15 lines)
- **TODO 2**: `compute_loss_free_running()` — Feed model's own predictions (~15 lines)
- **TODO 3**: `compute_loss_scheduled_sampling()` — Probabilistic mix of both (~18 lines)
- **TODO 4**: `sample_with_temperature()` — Temperature-scaled sampling (~5 lines)

```bash
uv run python subtask2_teacher_forcing.py
```

### Subtask 3: Full Model & Training (`subtask3_model.py` + `subtask3_train.py`)

Build and train the complete Fantasy Name Generator:

- **TODO 1** (model): `forward()` — Training with teacher forcing (~20 lines)
- **TODO 2** (model): `generate()` — Inference with temperature sampling (~18 lines)
- **TODO 1** (train): `train_step()` — Single training step (~12 lines)
- **TODO 2** (train): `generate_names_for_all_categories()` — Generate and display names (~10 lines)

```bash
# Test model shapes
uv run python subtask3_model.py

# Train (fast iteration)
uv run python subtask3_train.py --mini

# Train (full)
uv run python subtask3_train.py
```

## Expected Output

After training, the model generates names with distinct phonetic patterns per category:

| Category | Example Names |
|----------|--------------|
| Elf | Thalindra, Sereniel, Elowen |
| Dwarf | Grumnak, Thordin, Bazrik |
| Dragon | Zyraxoth, Vexithyr, Shaedrax |
| Fairy | Pixella, Twylana, Fleena |

## TODO Summary

11 TODOs, ~138 lines total:

| # | File | Function | Lines | Difficulty |
|---|------|----------|-------|-----------|
| 1 | subtask1 | `encode()` | ~5 | Easy |
| 2 | subtask1 | `decode_step()` | ~5 | Easy |
| 3 | subtask1 | `generate_greedy()` | ~15 | Medium |
| 4 | subtask2 | `compute_loss_with_teacher_forcing()` | ~15 | Medium |
| 5 | subtask2 | `compute_loss_free_running()` | ~15 | Medium |
| 6 | subtask2 | `compute_loss_scheduled_sampling()` | ~18 | Medium |
| 7 | subtask2 | `sample_with_temperature()` | ~5 | Easy |
| 8 | subtask3_model | `forward()` | ~20 | Hard |
| 9 | subtask3_model | `generate()` | ~18 | Hard |
| 10 | subtask3_train | `train_step()` | ~12 | Medium |
| 11 | subtask3_train | `generate_names_for_all_categories()` | ~10 | Easy |

# Twitter Emotion Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6+-792ee5.svg)](https://lightning.ai/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A BERT-based emotion classification model that detects six emotions (sadness,
joy, love, anger, fear, surprise) from text. Built with PyTorch Lightning for
clean, scalable training. Fine-tuned on the
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## Quick Demo

```bash
# Predict emotion from text
uv run python scripts/predict.py "I am so happy today!"
# Output: Emotion: joy (confidence: 95.32%)

uv run python scripts/predict.py "This makes me really angry"
# Output: Emotion: anger (confidence: 89.17%)
```

## Features

- **PyTorch Lightning** - Clean, organized training code with automatic
  mixed precision, checkpointing, and logging
- **Multi-GPU Support** - DDP, FSDP, and DeepSpeed strategies
- **TensorBoard Logging** - Real-time training visualization
- **Early Stopping** - Automatic training termination when validation plateaus
- **Class Balancing** - Oversampling for imbalanced emotion classes

## Installation

**Requirements:** Python 3.10-3.12, CUDA-capable GPU (recommended),
[uv](https://docs.astral.sh/uv/)

```bash
# Clone the repository
git clone https://github.com/gabe-zhang/twitter-emotion-classification.git
cd twitter-emotion-classification

# Install dependencies (uv creates venv automatically)
uv sync
```

## Usage

### Training

```bash
# Default training (20 epochs, batch size 4, mixed precision)
uv run python scripts/train.py

# Custom training parameters
uv run python scripts/train.py --epochs 10 --batch-size 8 --lr 1e-5

# Multi-GPU training (2 GPUs)
uv run python scripts/train.py --devices 2

# Use all available GPUs
uv run python scripts/train.py --devices -1

# Full precision training (if AMP causes issues)
uv run python scripts/train.py --precision 32

# Quick test run (1 batch)
uv run python scripts/train.py --fast-dev-run
```

Or use the shell script:

```bash
./scripts/train.sh                      # Default training
./scripts/train.sh --devices 2          # Multi-GPU training
./scripts/train.sh --fast-dev-run       # Quick test
```

### Cloud GPU Training

For cloud instances (AWS, GCP, Lambda Labs, etc.):

```bash
# Clone and setup
git clone https://github.com/gabe-zhang/twitter-emotion-classification.git
cd twitter-emotion-classification
uv sync

# Train with all available GPUs
uv run python scripts/train.py --devices -1 --epochs 20

# Monitor with TensorBoard
uv run tensorboard --logdir logs/
```

### Inference

```bash
# Single text
uv run python scripts/predict.py "I feel so lonely"

# Multiple texts
uv run python scripts/predict.py "I love you" "I hate this" "What a surprise!"

# From stdin
echo "Today is a great day" | uv run python scripts/predict.py --stdin
cat texts.txt | uv run python scripts/predict.py --stdin
```

### Configuration

Edit `configs/default.yaml` to customize hyperparameters:

```yaml
model:
  name: "bert-base-cased"
  num_classes: 6
  dropout: 0.5
  max_length: 512

training:
  batch_size: 4
  learning_rate: 5.0e-7
  epochs: 20
  num_workers: 2
  precision: "16-mixed"
  early_stopping_patience: 3
```

### CLI Options

```
--epochs          Number of training epochs (default: 20)
--batch-size      Training batch size (default: 4)
--lr              Learning rate (default: 5e-7)
--devices         Number of GPUs (-1 for all available)
--strategy        Parallelism: auto, ddp, fsdp, deepspeed
--precision       32, 16-mixed, or bf16-mixed
--num-workers     Data loading workers (default: 2)
--patience        Early stopping patience (default: 3)
--fast-dev-run    Quick test with 1 batch
--no-resample     Disable class balancing
--no-early-stopping  Disable early stopping
```

## Project Structure

```
twitter-emotion-classification/
├── src/
│   ├── config.py               # Configuration loader
│   ├── data/
│   │   └── dataset.py          # EmotionDataModule (Lightning)
│   ├── models/
│   │   └── bert_classifier.py  # BertClassifier (LightningModule)
│   └── training/
│       ├── trainer.py          # Lightning re-exports
│       └── utils.py            # Device utilities
├── scripts/
│   ├── train.py                # Training CLI
│   ├── predict.py              # Inference CLI
│   ├── train.sh                # Training shell script
│   └── evaluate.sh             # Evaluation shell script
├── notebooks/                  # Jupyter notebooks
├── configs/
│   └── default.yaml            # Hyperparameters
├── checkpoints/                # Model weights (.ckpt)
├── logs/                       # TensorBoard logs
└── pyproject.toml              # Dependencies (uv)
```

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 0.93 |
| Weighted F1 | 0.93 |

**Per-class Performance:**

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Sadness | 0.94 | 0.93 | 0.93 |
| Joy | 0.95 | 0.96 | 0.95 |
| Love | 0.86 | 0.89 | 0.87 |
| Anger | 0.93 | 0.90 | 0.91 |
| Fear | 0.91 | 0.89 | 0.90 |
| Surprise | 0.85 | 0.82 | 0.83 |

## Dataset

This project uses the **Emotion** dataset from HuggingFace:
- **Source:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- **Size:** ~20,000 English Twitter messages
- **Labels:** sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

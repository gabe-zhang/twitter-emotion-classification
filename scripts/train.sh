#!/bin/bash
# Train BERT emotion classifier with PyTorch Lightning
#
# Usage:
#   ./scripts/train.sh                    # Default training
#   ./scripts/train.sh --devices 2        # Multi-GPU training
#   ./scripts/train.sh --devices -1       # Use all GPUs
#   ./scripts/train.sh --epochs 10        # Custom epochs
#   ./scripts/train.sh --fast-dev-run     # Quick test (1 batch)
#   ./scripts/train.sh --no-resample      # Without class balancing
#   ./scripts/train.sh --precision 32     # Full precision (no AMP)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Twitter Emotion Classification Training ==="
echo "Using PyTorch Lightning"
echo ""

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

echo "Starting training..."
uv run python scripts/train.py "$@"

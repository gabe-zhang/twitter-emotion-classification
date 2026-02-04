#!/bin/bash
# Run emotion prediction on text
#
# Usage:
#   ./scripts/evaluate.sh "I am so happy today!"
#   ./scripts/evaluate.sh "text1" "text2" "text3"
#   echo "I feel sad" | ./scripts/evaluate.sh --stdin

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/evaluate.sh \"text to classify\""
    echo "       ./scripts/evaluate.sh --stdin  (read from stdin)"
    exit 1
fi

uv run python scripts/predict.py "$@"

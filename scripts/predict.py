#!/usr/bin/env python3
"""Run inference with trained emotion classifier using PyTorch Lightning.

Usage:
    python scripts/predict.py "I am so happy today!"
    python scripts/predict.py "This makes me angry" "I love this movie"
    echo "I feel sad" | python scripts/predict.py --stdin
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINT_DIR, EMOTION_LABELS, MAX_LENGTH, MODEL_NAME
from src.models.bert_classifier import BertClassifier
from transformers import BertTokenizer


def predict(
    texts: list[str],
    model: BertClassifier,
    tokenizer: BertTokenizer,
    device: torch.device,
) -> list[dict]:
    """Predict emotions for a list of texts.

    Args:
        texts: List of text strings to classify.
        model: Trained BertClassifier model.
        tokenizer: BERT tokenizer.
        device: Torch device.

    Returns:
        List of dicts with 'text', 'emotion', and 'confidence' keys.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            output = model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            results.append({
                "text": text,
                "emotion": EMOTION_LABELS[predicted_class],
                "confidence": confidence,
            })

    return results


def load_model(checkpoint_path: Path, device: torch.device) -> BertClassifier:
    """Load model from Lightning checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Loaded BertClassifier model.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load from Lightning checkpoint
    model = BertClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict emotions from text")
    parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to classify",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read texts from stdin (one per line)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.ckpt",
        help="Checkpoint filename (default: best_model.ckpt)",
    )
    args = parser.parse_args()

    # Get texts from args or stdin
    if args.stdin:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    elif args.texts:
        texts = args.texts
    else:
        parser.print_help()
        sys.exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Run predictions
    results = predict(texts, model, tokenizer, device)

    # Print results
    for result in results:
        print(f"Text: {result['text']}")
        emotion = result["emotion"]
        confidence = result["confidence"]
        print(f"  Emotion: {emotion} (confidence: {confidence:.2%})")
        print()


if __name__ == "__main__":
    main()

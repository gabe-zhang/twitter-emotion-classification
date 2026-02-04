#!/usr/bin/env python3
"""Train BERT classifier for emotion detection using PyTorch Lightning.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 10 --batch-size 8 --lr 1e-5
    python scripts/train.py --accelerator gpu --devices 2  # Multi-GPU
    python scripts/train.py --fast-dev-run  # Quick test run
"""

import argparse
import logging
import sys
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import BATCH_SIZE, CHECKPOINT_DIR, EPOCHS, LEARNING_RATE
from src.data.dataset import EmotionDataModule
from src.models.bert_classifier import BertClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train BERT emotion classifier with PyTorch Lightning"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Disable oversampling for class balance",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, cpu, tpu)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (-1 for all available)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Parallelism strategy (auto, ddp, fsdp, deepspeed)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision (default: 16-mixed for AMP)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development test (1 batch train/val/test)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)",
    )
    args = parser.parse_args()

    # Set up data module
    logger.info("Setting up data module...")
    data_module = EmotionDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resample=not args.no_resample,
    )

    # Set up model
    logger.info("Initializing model...")
    model = BertClassifier(learning_rate=args.lr)

    # Set up callbacks
    callbacks = [RichProgressBar()]

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_model",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if not args.no_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=args.patience,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Set up logger
    tb_logger = TensorBoardLogger(
        save_dir=CHECKPOINT_DIR.parent / "logs",
        name="emotion_classifier",
    )

    # Set up trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=callbacks,
        logger=tb_logger,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test with best model
    if not args.fast_dev_run:
        logger.info("Evaluating on test set...")
        trainer.test(model, data_module, ckpt_path="best")

        best_path = checkpoint_callback.best_model_path
        logger.info(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    main()

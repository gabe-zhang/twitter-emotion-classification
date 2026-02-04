"""Dataset utilities for emotion classification with PyTorch Lightning."""

from typing import Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from ..config import (
    BATCH_SIZE,
    DATASET_NAME,
    EMOTION_LABELS,
    LABEL_TO_ID,
    MAX_LENGTH,
    MODEL_NAME,
    RANDOM_SEED,
    RESAMPLE,
    TEST_SIZE,
    VAL_SIZE,
)


class EmotionDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for emotion classification with BERT tokenization.

    Args:
        df: DataFrame with 'text' and 'category' columns.
        tokenizer: BERT tokenizer instance.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.labels = [LABEL_TO_ID[label] for label in df["category"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        return self.texts[idx], np.array(self.labels[idx])

    def classes(self) -> List[int]:
        """Return all labels in the dataset."""
        return self.labels


class EmotionDataModule(L.LightningDataModule):
    """Lightning DataModule for emotion classification.

    Handles data loading, preprocessing, and DataLoader creation.

    Args:
        batch_size: Batch size for DataLoaders.
        num_workers: Number of workers for data loading.
        resample: Whether to apply oversampling to balance classes.
        test_size: Fraction of data for test set.
        val_size: Fraction of data for validation set.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        num_workers: int = 2,
        resample: bool = RESAMPLE,
        test_size: float = TEST_SIZE,
        val_size: float = VAL_SIZE,
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resample = resample
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        self.train_dataset: Optional[EmotionDataset] = None
        self.val_dataset: Optional[EmotionDataset] = None
        self.test_dataset: Optional[EmotionDataset] = None

    def prepare_data(self):
        """Download data if needed (called on single process)."""
        load_dataset(DATASET_NAME)

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for each stage (called on every process)."""
        if stage == "fit" or stage is None:
            train_df, val_df, _ = self._load_and_split_data()
            self.train_dataset = EmotionDataset(
                train_df, self.tokenizer, self.max_length
            )
            self.val_dataset = EmotionDataset(
                val_df, self.tokenizer, self.max_length
            )

        if stage == "test" or stage is None:
            _, _, test_df = self._load_and_split_data()
            self.test_dataset = EmotionDataset(
                test_df, self.tokenizer, self.max_length
            )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def _load_and_split_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split the emotion dataset."""
        dataset = load_dataset(DATASET_NAME)

        # Combine all splits
        X_train = dataset["train"][:]["text"]
        y_train = dataset["train"][:]["label"]
        X_val = dataset["validation"][:]["text"]
        y_val = dataset["validation"][:]["label"]
        X_test = dataset["test"][:]["text"]
        y_test = dataset["test"][:]["label"]

        X_all = np.concatenate((X_train, X_val, X_test), axis=0)
        y_all = np.concatenate((y_train, y_val, y_test), axis=0)

        # Create DataFrame with string labels
        data = pd.DataFrame({"category": y_all, "text": X_all})
        data["category"] = data["category"].replace(EMOTION_LABELS)

        # Split into train/val/test
        np.random.seed(RANDOM_SEED)
        df_train, df_temp = train_test_split(
            data,
            test_size=(self.test_size + self.val_size),
            random_state=RANDOM_SEED,
        )
        df_val, df_test = train_test_split(
            df_temp,
            test_size=self.test_size / (self.test_size + self.val_size),
            random_state=RANDOM_SEED,
        )

        # Apply oversampling to training data if requested
        if self.resample:
            df_train = self._resample_data(df_train)

        return df_train, df_val, df_test

    @staticmethod
    def _resample_data(df: pd.DataFrame) -> pd.DataFrame:
        """Apply random oversampling to balance class distribution."""
        X = np.array(df["text"]).reshape(-1, 1)
        y = df["category"]

        sampler = RandomOverSampler(
            sampling_strategy="not majority", random_state=RANDOM_SEED
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        return pd.DataFrame(
            {"text": X_resampled.flatten(), "category": y_resampled}
        )


def get_tokenizer() -> BertTokenizer:
    """Get the BERT tokenizer."""
    return BertTokenizer.from_pretrained(MODEL_NAME)

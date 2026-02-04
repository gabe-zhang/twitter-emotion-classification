"""BERT-based classifier for emotion detection using PyTorch Lightning."""

import lightning as L
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
from transformers import BertModel

from ..config import DROPOUT, LEARNING_RATE, MODEL_NAME, NUM_CLASSES


class BertClassifier(L.LightningModule):
    """BERT-based text classifier for emotion detection.

    Architecture:
        BERT Encoder -> Pooled Output (768 dim)
        -> Dropout -> Linear (768 -> num_classes)

    Args:
        dropout: Dropout probability for regularization.
        num_classes: Number of output classes.
        model_name: Pretrained BERT model name.
        learning_rate: Learning rate for optimizer.
    """

    def __init__(
        self,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        model_name: str = MODEL_NAME,
        learning_rate: float = LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="weighted"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_length).
            attention_mask: Attention mask tensor (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids, attention_mask, labels = self._unpack_batch(batch)
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids, attention_mask, labels = self._unpack_batch(batch)
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log(
            "val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_ids, attention_mask, labels = self._unpack_batch(batch)
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate
        )

    def on_train_start(self):
        """Ensure BERT is in train mode for fine-tuning."""
        self.bert.train()

    def _unpack_batch(self, batch):
        """Unpack batch from DataLoader."""
        batch_input, labels = batch
        input_ids = batch_input["input_ids"].squeeze(1)
        attention_mask = batch_input["attention_mask"].squeeze(1)
        labels = labels.long()
        return input_ids, attention_mask, labels

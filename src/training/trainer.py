"""Training module - now uses PyTorch Lightning.

This module is kept for backward compatibility. For training, use:
    - scripts/train.py for CLI training
    - src.models.bert_classifier.BertClassifier (LightningModule)
    - src.data.dataset.EmotionDataModule (LightningDataModule)

Example:
    from lightning import Trainer
    from src.models.bert_classifier import BertClassifier
    from src.data.dataset import EmotionDataModule

    model = BertClassifier()
    data = EmotionDataModule()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)
"""

# Re-export for backward compatibility
from lightning import Trainer

from ..data.dataset import EmotionDataModule
from ..models.bert_classifier import BertClassifier

__all__ = ["Trainer", "BertClassifier", "EmotionDataModule"]

"""Centralized configuration for Twitter Emotion Classification.

Loads settings from configs/default.yaml and provides Python constants.
"""

from pathlib import Path

import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

# Load configuration from YAML
_config_path = CONFIG_DIR / "default.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)

# Model configuration
MODEL_NAME: str = _config["model"]["name"]
NUM_CLASSES: int = _config["model"]["num_classes"]
DROPOUT: float = _config["model"]["dropout"]
MAX_LENGTH: int = _config["model"]["max_length"]

# Training configuration
BATCH_SIZE: int = _config["training"]["batch_size"]
LEARNING_RATE: float = _config["training"]["learning_rate"]
EPOCHS: int = _config["training"]["epochs"]
NUM_WORKERS: int = _config["training"]["num_workers"]
PRECISION: str = _config["training"]["precision"]
EARLY_STOPPING_PATIENCE: int = _config["training"]["early_stopping_patience"]

# Data configuration
DATASET_NAME: str = _config["data"]["dataset_name"]
RESAMPLE: bool = _config["data"]["resample"]
TEST_SIZE: float = _config["data"]["test_size"]
VAL_SIZE: float = _config["data"]["val_size"]
RANDOM_SEED: int = _config["data"]["random_seed"]

# Emotion label mappings
EMOTION_LABELS: dict[int, str] = {
    int(k): v for k, v in _config["labels"].items()
}
LABEL_TO_ID: dict[str, int] = {v: k for k, v in EMOTION_LABELS.items()}

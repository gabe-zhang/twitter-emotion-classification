"""Training utilities using PyTorch Lightning."""

from lightning import Trainer

from .utils import get_device

__all__ = ["Trainer", "get_device"]

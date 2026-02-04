"""Training utilities (kept for backward compatibility with notebooks)."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU).

    Also enables cudnn.benchmark for faster training when input sizes
    are consistent.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using CUDA device: {gpu_name}")
        logger.info(f"CUDA Memory: {gpu_mem:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

"""
Utility functions for device management and logging
"""
import torch
import logging
from typing import Optional, Union

def setup_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Setup compute device for inference"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
        
    if not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("CUDA device requested but CUDA is not available")
        
    return device

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("sparse_mor")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def profile_memory(prefix: str = "") -> dict:
    """Profile current GPU memory usage"""
    if not torch.cuda.is_available():
        return {}
        
    return {
        f"{prefix}allocated": torch.cuda.memory_allocated(),
        f"{prefix}max_allocated": torch.cuda.max_memory_allocated(),
        f"{prefix}reserved": torch.cuda.memory_reserved(),
        f"{prefix}max_reserved": torch.cuda.max_memory_reserved()
    }

def tensor_size_mb(tensor: torch.Tensor) -> float:
    """Get size of tensor in MB"""
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)

def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
import random

import numpy as np


def seed_tf(seed=0):
    """Set random seeds for reproducibility in TensorFlow.
    
    Args:
        seed: Random seed value (default: 0)
    """
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Configure GPU for deterministic operations (if available)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)


def seed_torch(seed):
    """Set random seeds for reproducibility in PyTorch.
    
    Args:
        seed: Random seed value
    """
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
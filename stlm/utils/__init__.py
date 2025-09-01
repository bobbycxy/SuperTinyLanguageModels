from .hf_loader import get_hf_model
from .dist_utils import *
from .lora import *

__all__ = [
    "get_hf_model",
    "get_transformer_layer_classes"
]

def count_parameters(model, verbose=False):
    """Count total and trainable parameters, excluding tied weights."""
    total, trainable = 0, 0

    for name, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    
    return total, trainable
# stlm/cores/components/standard.py

from stlm import BaseCore
import torch.nn as nn
from stlm.registry import register_component
from .components import TransformerBlock

@register_component("core", "standard")
class StandardCore(BaseCore):
    def __init__(self, model_cfg, checkpointing=False):
        super().__init__()
        hidden_size = model_cfg["embedder"]["hidden_size"]
        num_layers = model_cfg["core"]["num_layers"]

        self.layers = nn.ModuleList([
            TransformerBlock(model_cfg, checkpointing=checkpointing)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_size)
                                       
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return self.layer_norm(x)
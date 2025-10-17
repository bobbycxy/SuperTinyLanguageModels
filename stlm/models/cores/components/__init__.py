# stlm/cores/components/__init__.py

import torch
import torch.nn as nn
from stlm.registry import REGISTRY
import pkgutil, importlib, pathlib

package_dir = pathlib.Path(__file__).resolve().parent
for module_info in pkgutil.walk_packages([str(package_dir)], prefix=__name__ + "."):
    importlib.import_module(module_info.name)

class TransformerBlock(nn.Module):
    def __init__(self, model_cfg, checkpointing=False):
        super().__init__()
        self.checkpointing = checkpointing
        hidden_size = model_cfg["embedder"]["hidden_size"]

        attention_cls = REGISTRY["attention"][model_cfg["core"]["attention"]["name"]]
        ffn_cls = REGISTRY["ffn"][model_cfg["core"]["ffn"]["name"]]

        self.attention = attention_cls(model_cfg)
        self.ffn = ffn_cls(model_cfg)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward_fn(self, x, attn_mask=None):
        h = x + self.attention(self.layer_norm1(x), attn_mask)
        h = h + self.ffn(self.layer_norm2(h))
        return h
    
    def forward(self, x, attn_mask=None):
        if self.checkpointing:
            return torch.utils.checkpoint.checkpoint(self.forward_fn, x, attn_mask, use_reentrant=False)
        else:
            return self.forward_fn(x, attn_mask)
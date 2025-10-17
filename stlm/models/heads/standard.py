# stlm/heads/standard.py

import torch.nn as nn
from stlm import BaseHead
from stlm.registry import register_component

@register_component("head", "standard")
class StandardLMHead(BaseHead):
    def __init__(self, model_cfg, embedder=None):
        super().__init__()
        hidden_size = model_cfg["embedder"]["hidden_size"]
        vocab_size = model_cfg["tokenizer"]["vocab_size"]
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        if model_cfg["head"].get("tie_weights", False) and embedder is not None:
            self.head.weight = embedder.token_embedding.weight
            self.head.weight.requires_grad = True
    
    def forward(self, x):
        return self.head(x)

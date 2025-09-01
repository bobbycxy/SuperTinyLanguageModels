# stlm/cores/components/ffn/standard.py

import torch.nn as nn
import torch.nn.functional as F
from stlm import register_component

@register_component("ffn", "standard")
class StandardFFN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.hidden_size = model_cfg["embedder"]["hidden_size"]
        self.intermediate_size = model_cfg["core"]["ffn"]["intermediate_size"]
        self.checkpointing = False

        self.fully_connected1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fully_connected2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(model_cfg["core"]["ffn"]["dropout"])

        activation = model_cfg["core"]["ffn"]["activation"]
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"This activation {activation} has not yet been added.")
        
    def forward(self, x):
        h = self.fully_connected1(x)
        h = self.activation(h)
        h = self.dropout(h)
        out = self.fully_connected2(h)
        out = self.dropout(out)
        return out

# stlm/cores/components/attention/standard.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stlm.registry import register_component

@register_component("attention", "standard")
class StandardAttention(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.hidden_size = model_cfg["embedder"]["hidden_size"]
        self.num_heads = model_cfg["core"]["attention"]["num_heads"]
        assert self.hidden_size % self.num_heads == 0, "hidden size must be divisible by num_heads"
        
        self.dropout = model_cfg["core"]["attention"]["dropout"]
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_dropout = nn.Dropout(self.dropout)
        self.residual_dropout = nn.Dropout(self.dropout)

    def forward(self, x, attn_mask=None):
        B, T, H = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hd]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hd]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, nh, T, hd]

        attention_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale

        if attn_mask is not None:
            attn_mask = attn_mask.contiguous().view(B, 1, 1, T)
            attention_scores = attention_scores.masked_fill(attn_mask==0, float("-inf"))

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, v) # [B, nh, T, hd]
        context = context.transpose(1, 2).contiguous().view(B, T, H) # [B, T, H]

        out = self.o_proj(context)
        out = self.residual_dropout(out)

        return out

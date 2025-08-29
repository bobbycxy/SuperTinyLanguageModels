# stlm/core.py

import torch
import torch.nn as nn
from stlm.registry import REGISTRY
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTokenizer(ABC):
    pad_token_id: int
    eos_token_id: int

    @abstractmethod
    def encode(self, text: str, add_eos: bool = True, max_length: int = None) -> List[int]:
        pass

    @abstractmethod
    def batch_encode(self, texts: List[str], add_eos: bool = True, max_length: int = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass

    @abstractmethod
    def batch_decode(self, batch_token_ids: List[List[int]]) -> List[str]:
        pass

    @abstractmethod
    def pad_batch(self, batch_token_ids: List[List[int]], direction: str = "right") -> Dict[str, Any]:
        pass


class BaseEmbedder(nn.Module, ABC):
    @abstractmethod
    def forward(self, token_ids):
        pass

class BaseCore(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, attn_mask=None):
        pass

class BaseHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

class STLM(nn.Module):
    def __init__(self, embedder: nn.Module, core: nn.Module, head: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.core = core
        self.head = head

    def forward(self, token_ids, attn_mask=None):
        embeddings = self.embedder(token_ids)            # [B, T, H]
        hidden_state = self.core(embeddings, attn_mask)  # [B, T, H]
        logits = self.head(hidden_state)                 # [B, T, V]
        return logits



class BaseTrainer(ABC):
    def __init__(self, model, optimizer, scheduler=None, device="cuda", grad_accum_steps=1, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = grad_accum_steps

        if use_mixed_precision and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
            self.autocast = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
            self.autocast = torch.amp.autocast(enabled=False)

    @abstractmethod
    def train_step(self, batch, step=None):
        pass

    # @abstractmethod
    # def eval_step(self, batch):
    #     pass

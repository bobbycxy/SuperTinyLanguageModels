# stlm/trainer/base.py
from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    def __init__(self, model, optimizer, scheduler=None, device="cuda", grad_accum_steps=1, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = grad_accum_steps

        # Mixed precision setup
        if use_mixed_precision and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
            self.autocast = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
            self.autocast = torch.cuda.amp.autocast(enabled=False)

    @abstractmethod
    def train_step(self, batch, step=None):
        pass

    @abstractmethod
    def eval_step(self, batch):
        pass

    def maybe_cleanup(self, step=None, interval=100):
        """
        Utility: clear GPU cache periodically
        """
        if step is not None and step % interval == 0:
            torch.cuda.empty_cache()

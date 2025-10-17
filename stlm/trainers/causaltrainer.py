import torch
import torch.nn as nn
from stlm.core import BaseTrainer, TrainerState, LossOutput
import torch.distributed as dist
import wandb
import math

class CausalTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss_and_metrics(self, batch) -> LossOutput:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with self.autocast:
            logits = self.model(input_ids)
            B, T, V = logits.size()
            loss = self.loss_fn(logits.view(B * T, V), labels.view(B * T))
        
        token_count = (labels != -100).sum().item()
        metrics = {"cross_entropy": loss.item()}
        return LossOutput(loss=loss, token_count=token_count, metrics=metrics)
    
    
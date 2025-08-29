# stlm/trainer/sft_trainer.py
import torch
import torch.nn.functional as F
from stlm.trainer.base import BaseTrainer

class SFTTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler=None, device="cuda", grad_accum_steps=1, use_mixed_precision=True):
        super().__init__(model, optimizer, scheduler, device) 
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

    def train_step(self, batch, step=None):
        """
        Train step with gradient accumulation & mixed precision.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_seq_len = 0

        for micro_step in range(self.grad_accum_steps):
            input_ids, labels, attn_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            attn_mask = attn_mask.to(self.device, non_blocking=True)

            with self.autocast:
                logits = self.model(input_ids, attn_mask=attn_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss = loss / self.grad_accum_steps

            # backward pass
            self.scaler.scale(loss).backward()
            total_loss += loss.item()
            total_seq_len += attn_mask.sum(dim=-1).float().mean().item()

            # free step-local tensors early
            del input_ids, labels, attn_mask, logits, loss

        # optimizer update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler:
            self.scheduler.step()

        # optional memory cleanup (every N steps)
        torch.cuda.empty_cache()

        avg_seq_len = total_seq_len / self.grad_accum_steps
        return {"loss": total_loss, "avg_seq_len": avg_seq_len}

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad(), self.autocast:
            input_ids, labels, attn_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            attn_mask = attn_mask.to(self.device, non_blocking=True)

            logits = self.model(input_ids, attn_mask=attn_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            del input_ids, labels, attn_mask, logits
            torch.cuda.empty_cache()

        return {"loss": loss.item()}

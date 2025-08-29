import torch
import torch.nn as nn
from stlm.core import BaseTrainer
import torch.distributed as dist
import wandb

class SFTTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler=None, device="cuda",
                 grad_accum_steps=1, use_mixed_precision=True):
        super().__init__(model, optimizer, scheduler, device,
                         grad_accum_steps, use_mixed_precision)
        self.model = model.to(device)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        # tracking
        self._step_loss = 0.0
        self._step_count = 0
        self._token_count = 0
        self.processed_tokens = 0
        self.processed_steps = 0

    def train_step(self, batch, step=None):
        """
        Perform one training step (with gradient accumulation support).
        Args:
            batch: tuple (input_ids, labels, attn_mask)
        Returns:
            metrics dict
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        with self.autocast:
            logits = self.model(input_ids, attn_mask)  # [B, T, V]
            loss = self.loss_function(
                logits.view(-1, logits.size(-1)),  # [B*T, V]
                labels.view(-1)                    # [B*T]
            ) / self.grad_accum_steps  # normalize loss for accum

        self.scaler.scale(loss).backward()

        # track metrics
        self._step_loss += loss.item() * self.grad_accum_steps
        self._step_count += 1

        self._token_count += attn_mask.sum().item()

        return {"loss": self._step_loss / self._step_count}

    def optimizer_step(self, step=None, epoch=None):
        # optimizer update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

        # average loss across accumulation
        if self._step_count > 0:
            avg_loss = self._step_loss / self._step_count
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_mean(loss_tensor).item()

            # reduce token count across GPUs
            if dist.is_initialized():
                token_tensor = torch.tensor(self._token_count, device=self.device, dtype=torch.long)
                dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
                total_tokens = token_tensor.item()
            else:
                total_tokens = self._token_count

            # update global counters
            self.processed_steps += 1
            self.processed_tokens += int(total_tokens)

            # log only on rank 0
            if not dist.is_initialized() or dist.get_rank() == 0:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/processed_tokens": self.processed_tokens,
                    "train/processed_steps": self.processed_steps,
                    "epoch": epoch,
                }, step=self.processed_tokens)

                print(f"[TRAIN] step={self.processed_steps} tokens={self.processed_tokens} loss={avg_loss:.4f}")

        # reset trackers
        self._step_loss = 0.0
        self._step_count = 0
        self._token_count = 0

    def eval_step(self, batch):
        """Run a forward pass on validation data, return avg loss."""
        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.autocast:
                logits = self.model(input_ids, attn_mask)
                loss = self.loss_function(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

        loss_tensor = torch.tensor(loss.item(), device=self.device)
        avg_loss = reduce_mean(loss_tensor).item()

        return avg_loss

    

def reduce_mean(tensor):
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

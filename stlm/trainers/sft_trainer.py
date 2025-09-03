import torch
import torch.nn as nn
from stlm.core import BaseTrainer
import torch.distributed as dist
import wandb
import math

class SFTTrainer(BaseTrainer):
    def __init__(self, model, optimizer,
                 train_dataloader, val_dataloader,
                 scheduler=None, device="cuda",
                 grad_accum_steps=1, use_mixed_precision=True):
        super().__init__(model, optimizer, train_dataloader,
                         val_dataloader, scheduler, device,
                         grad_accum_steps, use_mixed_precision)
        self.model = model.to(device)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)

        self.processed_tokens = 0
        self._current_epoch = 0
        self.train_iter = self._reset_iterator()
    
    def _reset_iterator(self):
        if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self._current_epoch)
        return iter(self.train_dataloader)

    def train_step(self, step):
        self.model.train()

        # step tracking
        self._step_loss, self._token_count = 0.0, 0

        for _ in range(self.grad_accum_steps):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self._current_epoch += 1
                self.train_iter = self._reset_iterator()
                batch = next(self.train_iter)

            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with self.autocast:
                logits = self.model(input_ids, attn_mask)
                micro_loss = self.loss_function(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

            # scale for backward
            loss = micro_loss / self.grad_accum_steps
            self.scaler.scale(loss).backward()

            # accumulate true loss * tokens
            valid_tokens = (labels != -100).sum().item()
            self._step_loss += micro_loss.item() * valid_tokens
            self._token_count += valid_tokens

        self.optimizer_step(step)

    def optimizer_step(self, step):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

        # effective batch loss (per token)
        avg_loss = self._step_loss / self._token_count
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = reduce_mean(loss_tensor).item()

        # perplexity
        avg_ppl = math.exp(avg_loss)

        # reduce tokens
        if dist.is_initialized():
            token_tensor = torch.tensor(self._token_count, device=self.device, dtype=torch.long)
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            total_tokens = token_tensor.item()
        else:
            total_tokens = self._token_count

        # update counters
        self.processed_tokens += int(total_tokens)

        if (not dist.is_initialized() or dist.get_rank() == 0) and wandb.run is not None:
            wandb.log({
                "train/loss": avg_loss,
                "train/perplexity": avg_ppl,
                "train/processed_tokens": self.processed_tokens,
                "train/steps": step,
                "train/epoch": self._current_epoch
            }, step=self.processed_tokens)

            print(f"[TRAIN] step={step} tokens={self.processed_tokens} loss={avg_loss:.4f} ppl={avg_ppl:.2f}")

    def eval_step(self, step):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with self.autocast:
                    logits = self.model(input_ids, attn_mask)
                    micro_loss = self.loss_function(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                valid_tokens = (labels != -100).sum().item()
                total_loss += micro_loss.item() * valid_tokens
                total_tokens += valid_tokens

        mean_loss = total_loss / total_tokens
        loss_tensor = torch.tensor(mean_loss, device=self.device)
        avg_loss = reduce_mean(loss_tensor).item()

        # perplexity
        avg_ppl = math.exp(avg_loss)
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and wandb.run is not None:
            wandb.log({
                "val/loss": avg_loss,
                "val/perplexity": avg_ppl,
                "val/steps": step,
            }, step=self.processed_tokens)

            print(f"[VAL] step={step} loss={avg_loss:.4f} ppl={avg_ppl:.2f}")


    

def reduce_mean(tensor):
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

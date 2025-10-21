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
    
    def train_step(self, step: int):
        self.model.train()
        step_loss, step_tokens = 0.0, 0

        for _ in range(self.grad_accum_steps):
            batch = self.next_batch()
            out = self.compute_loss_and_metrics(batch)
            self.backward(out.loss)
            
            step_loss += out.loss.item() * out.token_count
            step_tokens += out.token_count
            self.state.extra.update(out.metrics)

        self.optimizer_step()

        # average loss across GPUs
        avg_loss = step_loss / step_tokens
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        avg_loss = self.distributed_mean(avg_loss_tensor).item()
        avg_ppl = math.exp(avg_loss)

        # update state
        self.state.step = step
        self.state.loss = avg_loss
        self.state.perplexity = avg_ppl
        self.state.tokens += step_tokens

        # subclass handles logging
        self.log_train()

    def eval_step(self, step: int, prompt: str = None, max_new_tokens: int = 50):
        """Run a single validation step (does not modify training state)."""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        step_loss, step_tokens = 0.0, 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                out = self.compute_loss_and_metrics(batch)
                step_loss += out.loss.item() * out.token_count
                step_tokens += out.token_count

            # ====== Average loss across GPUs (same convention as train_step) ======
            avg_loss = step_loss / (step_tokens + 1e-8)
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = self.distributed_mean(avg_loss_tensor).item()
            avg_ppl = math.exp(avg_loss)

            # ====== Rank-0 optional text generation ======
            generated_text = None
            if not dist.is_initialized() or dist.get_rank() == 0:
                if prompt is not None:
                    generated_text = self.model.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                    )

        eval_state = TrainerState(
            step=step,
            epoch=self.state.epoch,  # reference current epoch, but not modify it
            tokens=step_tokens,       # independent of train token count
            loss=avg_loss,
            perplexity=avg_ppl,
            extra={"generated_text": generated_text} if generated_text is not None else {},
        )

        self.log_eval(eval_state)

        if dist.is_initialized():
            dist.barrier()

        return eval_state
    
    
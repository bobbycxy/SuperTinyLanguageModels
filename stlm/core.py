# stlm/core.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from stlm.registry import REGISTRY
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import contextlib
import math
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import wandb

@dataclass
class TrainerState:
    step: int = 0
    epoch: int = 0
    tokens: int = 0
    loss: float = 0.0
    perplexity: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self, prefix: str = "train") -> Dict[str, Any]:
        log_dict = {
            f"{prefix}/step": self.step,
            f"{prefix}/epoch": self.epoch,
            f"{prefix}/tokens": self.tokens,
            f"{prefix}/loss": self.loss,
            f"{prefix}/perplexity": self.perplexity,
        }
        for k, v in self.extra.items():
            log_dict[f"{prefix}/{k}"] = v
        return log_dict

@dataclass
class LossOutput:
    """Container for loss, normalization count, and metrics."""
    loss: torch.Tensor
    token_count: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def asdict(self):
        d = {"loss": self.loss.item(), "token_count": self.token_count}
        d.update(self.metrics)
        return d

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
    """Super Tiny Language Model architecture."""
    def __init__(self, embedder: nn.Module, core: nn.Module, head: nn.Module, tokenizer):
        super().__init__()
        self.embedder = embedder
        self.core = core
        self.head = head
        self.tokenizer = tokenizer

    def forward(self, token_ids, attn_mask=None):
        x = self.embedder(token_ids)
        x = self.core(x, attn_mask)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=50, eos_token_id=None):
        """Simple autoregressive generation (no distributed logic)."""
        self.eval()
        device = next(self.parameters()).device
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_eos=False)], device=device)

        for _ in range(max_new_tokens):
            input_ids = input_ids[:, -self.embedder.max_position_embeddings:]
            logits = self(input_ids)[:, -1, :] / temperature
            if top_k:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id and (next_token == eos_token_id).any():
                break

        return self.tokenizer.decode(input_ids[0].tolist())

class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader,
        val_dataloader=None,
        scheduler=None,
        device: str = "cuda",
        grad_accum_steps: int = 1,
        use_mixed_precision: bool = True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.state = TrainerState()

        if use_mixed_precision and torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
            self.autocast = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
            self.autocast = contextlib.nullcontext()

        self._train_iter = iter(self.train_dataloader)

    @abstractmethod
    def compute_loss_and_metrics(self, batch) -> LossOutput:
        pass

    @abstractmethod
    def log_train(self):
        pass

    @abstractmethod
    def log_eval(self):
        pass

    def next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self.state.epoch += 1
            self.train_dataloader.sampler.set_epoch(self.state.epoch)
            self._train_iter = iter(self.train_dataloader)
            return next(self._train_iter)
    
    def backward(self, loss: torch.Tensor):
        scaled_loss = loss / self.grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

    def optimizer_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

    @staticmethod
    def distributed_mean(t: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return t
        rt = t.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt


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

    
    def log_train(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if wandb.run is not None:
                wandb.log(
                    self.state.to_log_dict(prefix="train"),
                    step=self.state.tokens
                )
            print(
                f"[TRAIN] step {self.state.step} | tokens {self.state.tokens} | epoch {self.state.epoch} | loss {self.state.loss:.4f} | ppl {self.state.perplexity:.2f}"
            )

    def log_eval(self, eval_state: TrainerState):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if wandb.run is not None:
                wandb.log(eval_state.to_log_dict(prefix="eval"), step=self.state.tokens)

            # === Write eval outputs progressively to a text-based CSV ===
            csv_path = "eval_generated_texts.csv"
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

            # Prepare row data
            generated_text = eval_state.extra.get("generated_text", "")
            truncated_text = " ".join(generated_text.split()[:100]).replace("\n", " ")
            row = f"{eval_state.step},{eval_state.loss:.4f},{eval_state.perplexity:.2f},\"{truncated_text}\"\n"

            # Write header if file is new
            if not os.path.exists(csv_path):
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("step,loss,perplexity,generated_text\n")

            # Append new row
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(row)

            print(
                f"[EVAL] step {eval_state.step} | loss {eval_state.loss:.4f} | ppl {eval_state.perplexity:.2f} | generated_text: {generated_text}"
            )
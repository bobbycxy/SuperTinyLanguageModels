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

import logging
log = logging.getLogger(__name__)

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

    @classmethod
    def from_config(cls, cfg: dict):
        from stlm.tokenizers import build_tokenizer
        model_cfg = cfg["model"]
        checkpointing = cfg.get("trainer", {}).get("checkpointing", False)

        # Ensure all components exist in registry
        try:
            embedder_cls = REGISTRY["embedder"][model_cfg["embedder"]["name"]]
            core_cls     = REGISTRY["core"][model_cfg["core"]["name"]]
            head_cls     = REGISTRY["head"][model_cfg["head"]["name"]]
        except KeyError as e:
            raise KeyError(f"[build_from_config] Missing component in REGISTRY: {e}")

        # Tokenizer
        tokenizer = build_tokenizer(cfg)

        # Instantiate submodules
        embedder = embedder_cls(model_cfg=model_cfg, checkpointing=checkpointing, pad_token_id=tokenizer.pad_token_id)
        core = core_cls(model_cfg=model_cfg, checkpointing=checkpointing)
        head = head_cls(model_cfg=model_cfg, embedder=embedder if model_cfg["head"].get("tie_weights", False) else None)

        # Combine into the full model
        model = STLM(embedder, core, head, tokenizer)

        # Print parameter summary
        from stlm.utils import count_parameters
        total_params, trainable_params = count_parameters(model)
        print(f"[Model] Total params: {total_params:,} | Trainable: {trainable_params:,}")

        return model

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
        use_mixed_precision: bool = True,
        out_dir: str = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_sampler = getattr(train_dataloader, "sampler", None)
        self.val_dataloader = val_dataloader
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.state = TrainerState()
        self.out_dir = out_dir

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
            if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(self.state.epoch)
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
            log.info(
                f"[TRAIN] step {self.state.step} | tokens {self.state.tokens} | "
                f"epoch {self.state.epoch} | loss {self.state.loss:.4f} | ppl {self.state.perplexity:.2f}"
            )

    def log_eval(self, eval_state: TrainerState):
        # Only rank 0 logs to file / wandb
        if not dist.is_initialized() or dist.get_rank() == 0:
            if wandb.run is not None:
                wandb.log(eval_state.to_log_dict(prefix="eval"), step=self.state.tokens)

            # === Progressive CSV writing inside outputs dir ===
            csv_path = os.path.join(self.out_dir or ".", "eval_generated_texts.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            generated_text = eval_state.extra.get("generated_text", "")
            truncated_text = " ".join(generated_text.split()[:100]).replace("\n", " ")
            row = f"{eval_state.step},{eval_state.loss:.4f},{eval_state.perplexity:.2f},\"{truncated_text}\"\n"

            # Write header if file is new
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("step,loss,perplexity,generated_text\n")
                f.write(row)

            log.info(
                f"[EVAL] step {eval_state.step} | loss {eval_state.loss:.4f} | "
                f"ppl {eval_state.perplexity:.2f} | generated_text: {truncated_text}"
            )
    
    def save_checkpoint(self, step: int, filename: str = None):
        """Save model, optimizer, and trainer state (safe for DDP/FSDP)."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return  # only rank 0 saves

        ckpt_dir = self.out_dir or "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = filename or os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt")

        # Unwrap model for saving
        model_to_save = self.model
        if hasattr(model_to_save, "module"):  # DDP
            model_to_save = model_to_save.module
        elif hasattr(model_to_save, "model"):  # DDPWrapper or FSDPWrapper
            model_to_save = model_to_save.model
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module

        # Handle FSDP unwrapping safely
        if isinstance(self.model, torch.distributed.fsdp.FullyShardedDataParallel):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.state_dict_type(self.model, state_dict_type="full_state_dict"):
                state_dict = self.model.state_dict()
        else:
            state_dict = model_to_save.state_dict()

        # Package everything to save
        save_obj = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_state": asdict(self.state),
        }
        torch.save(save_obj, ckpt_path)
        log.info(f"Saved checkpoint to {ckpt_path}")

    def load_checkpoint(self, ckpt_path: str, strict: bool = True):
        """Load model, optimizer, and trainer state (works pre- or post-wrap)."""
        if not os.path.exists(ckpt_path):
            log.warning(f"Checkpoint not found: {ckpt_path}")
            return False

        map_location = "cpu" if not torch.cuda.is_available() else "cuda"
        checkpoint = torch.load(ckpt_path, map_location=map_location)

        # Unwrap model if wrapped
        model_to_load = self.model
        if hasattr(model_to_load, "module"):
            model_to_load = model_to_load.module
        elif hasattr(model_to_load, "model"):
            model_to_load = model_to_load.model
            if hasattr(model_to_load, "module"):
                model_to_load = model_to_load.module

        missing, unexpected = model_to_load.load_state_dict(
            checkpoint["model_state_dict"], strict=strict
        )
        log.info(f"Loaded model weights from {ckpt_path}")
        if missing or unexpected:
            log.warning(f"Missing keys: {missing}, Unexpected keys: {unexpected}")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "trainer_state" in checkpoint:
            self.state = TrainerState(**checkpoint["trainer_state"])

        return True
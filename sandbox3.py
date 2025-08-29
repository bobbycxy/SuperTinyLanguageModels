# sandbox3.py

import os
import signal
import sys
import torch
import torch.distributed as dist
import yaml

import stlm
from stlm.registry import REGISTRY


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_signal_handlers(rank):
    """Setup signal handlers for clean shutdown."""
    def shutdown_handler(signum, frame):
        print(f"[Rank {rank}] Received signal {signum}. Exiting.")
        if dist.is_initialized():
            dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def main():
    # Load config
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # torchrun provides these env vars
    rank = int(os.environ.get("RANK", 0))             # global rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # local GPU id
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- Distributed setup ----
    if world_size > 1 and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)  # set device first
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
        setup_signal_handlers(rank)
        dist_mode = True

    else:
        if world_size > 1:
            print("⚠️ Multiple GPUs detected but not using torchrun → running in single mode.")
        dist_mode = False

    # ---- Build model ----
    model = stlm.build_from_config(cfg)
    model = stlm.FSDPWrapper(model, device=device)

    # ---- Count trainable params ----
    trainable_params = count_trainable_params(model)
    print(f"[Rank {rank}] Trainable params: {trainable_params:,}")

    # ---- Cleanup ----
    if dist_mode and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("All ranks finished counting params. Exiting.")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

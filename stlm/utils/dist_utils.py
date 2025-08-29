# stlm/utils/dist_utils.py

import os
import torch
import torch.distributed as dist
import datetime

def get_transformer_layer_classes(model):
    """
    Dynamically find all transformer layer classes in the core model.
    In STLM, the main body lives in `model.core`.
    """
    layer_classes = set()

    # unwrap to core if available
    core_model = getattr(model, "core", model)

    for module in core_model.modules():
        cls = type(module)
        if cls.__name__.lower().endswith("decoderlayer") \
           or cls.__name__.lower().endswith("encoderlayer") \
           or cls.__name__.lower().endswith("block") \
           or cls.__name__.lower().endswith("layer"):
            layer_classes.add(cls)

    return layer_classes

def init_distributed_setup():
    """Initialize distributed environment (for torchrun)."""
    if dist.is_initialized():
        return  # already done

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600)
    )

    torch.cuda.set_device(local_rank)
    print(f"[Rank {rank}] Distributed setup complete "
          f"(local_rank={local_rank}, world_size={world_size})")

def is_dist():
    """Check if distributed process group is initialized."""
    return dist.is_initialized()
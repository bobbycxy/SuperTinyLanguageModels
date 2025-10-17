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
        if cls.__name__.lower().endswith(("decoderlayer", "encoderlayer", "block", "layer")):
            layer_classes.add(cls)

    return layer_classes


def init_distributed_setup():
    """
    Initialize torch.distributed for torchrun-style multi-GPU jobs.

    Returns:
        rank (int): global process rank
        world_size (int): total number of processes
        local_rank (int): process-local GPU index (device_id)
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank

    # read environment vars set by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # default master addr/port
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    # initialize process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Distributed setup complete (world_size={world_size})")

    return rank, world_size, local_rank


def is_dist():
    """Check if distributed process group is initialized."""
    return dist.is_initialized()

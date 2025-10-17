# stlm/wrappers/dist_wrapper.py
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import contextlib
from abc import ABC, abstractmethod
from stlm.utils.dist_utils import get_transformer_layer_classes


# -------------------- Base Wrapper --------------------
class BaseDistributedWrapper(nn.Module, ABC):
    """Base class for DDP/FSDP wrappers. Handles forwarding and attribute passthrough."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # store the model safely using __setattr__ to avoid recursion
        object.__setattr__(self, "model", model)

    def forward(self, *args, **kwargs):
        # Directly delegate forward to the wrapped model
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        """
        Safely delegate attribute access to the wrapped model.
        Handles nested DDP/FSDP modules by checking .module if needed.
        """
        model = object.__getattribute__(self, "model")

        # Try the wrapper itself first
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        # Try the model directly
        try:
            return getattr(model, name)
        except AttributeError:
            pass

        # If it's a DDP/FSDP model, delegate to .module
        if hasattr(model, "module"):
            try:
                return getattr(model.module, name)
            except AttributeError:
                pass

        raise AttributeError(f"{name} not found in wrapped model or wrapper.")

        
    def parameters(self, recurse: bool = True):
        """Expose wrapped model parameters."""
        model = object.__getattribute__(self, "model")
        return model.parameters(recurse)

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Must handle inference-safe parameter access for distributed modes."""
        pass


# -------------------- DDP Wrapper --------------------
class DDPWrapper(BaseDistributedWrapper):
    def __init__(self, model: nn.Module, device_id: int):
        ddp_model = DDP(
            model.to(f"cuda:{device_id}"),
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )
        super().__init__(ddp_model)

    def generate(self, *args, **kwargs):
        """DDP forwards directly to inner module (only rank 0 should call generate)."""
        return self.model.module.generate(*args, **kwargs)


# -------------------- FSDP Wrapper --------------------
class FSDPWrapper(BaseDistributedWrapper):
    def __init__(self, model: nn.Module, device_id: int):
        layer_classes = get_transformer_layer_classes(model)
        auto_wrap_policy = transformer_auto_wrap_policy(layer_classes)
        fsdp_model = FSDP(
            model.to(f"cuda:{device_id}"),
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device_id,
            use_orig_params=True,
        )
        super().__init__(fsdp_model)

    def generate(self, *args, **kwargs):
        """
        FSDP generation safely gathers full parameters within the context.
        Only rank 0 should call generate().
        """
        with FSDP.summon_full_params(self.model, recurse=True):
            return self.model(*args, **kwargs)

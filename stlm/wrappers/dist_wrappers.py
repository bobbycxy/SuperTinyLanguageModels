# stlm/wrappers/dist_wrapper.py

from functools import partial
from abc import ABC, abstractmethod
from stlm.utils import get_transformer_layer_classes
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

class BaseDistributedWrapper(ABC):
    def __init__(self, model, device):
        self.original_model = model.to(device)
        self.device = device
        self.wrapped_model = self._wrap_model()

    @abstractmethod
    def _wrap_model(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.wrapped_model, name)
    
class DDPWrapper(BaseDistributedWrapper):
    def _wrap_model(self):
        return DDP(self.original_model, device_ids=[self.device.index], output_device=self.device, find_unused_parameters=False)

class FSDPWrapper(BaseDistributedWrapper):
    def _wrap_model(self):
        transformer_layer_classes = get_transformer_layer_classes(self.original_model)
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_classes)
        return FSDP(
            self.original_model,
            device_id=self.device,   # torch.device('cuda:X')
            auto_wrap_policy=auto_wrap_policy,
            sync_module_states=True,
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            forward_prefetch=True
        )

import pkgutil
import importlib
import pathlib

from .core import STLM, BaseEmbedder, BaseCore, BaseHead, BaseTrainer, BaseTokenizer
from .registry import REGISTRY, register_component
from .tokenizers import build_tokenizer
from .wrappers import DDPWrapper, FSDPWrapper, LoRAWrapper
from .trainers.sft_trainer import SFTTrainer
from .utils import count_parameters

# --- Auto-import all submodules so @register_component decorators run ---
package_dir = pathlib.Path(__file__).resolve().parent
for module_info in pkgutil.walk_packages([str(package_dir)], prefix="stlm."):
    # Skip __init__.py itself to avoid recursion
    if module_info.name != __name__:
        importlib.import_module(module_info.name)

def build_from_config(cfg: dict):
    """Build an STLM model (optionally with wrappers) from config."""
    embedder_cls = REGISTRY["embedder"][cfg["model"]["embedder"]["name"]]
    core_cls     = REGISTRY["core"][cfg["model"]["core"]["name"]]
    head_cls     = REGISTRY["head"][cfg["model"]["head"]["name"]]

    # Global toggle
    checkpointing = cfg.get("trainer", {}).get("checkpointing", False)
    tokenizer = build_tokenizer(cfg)

    # Build components
    embedder = embedder_cls(
        model_cfg=cfg["model"],
        checkpointing=checkpointing,
        pad_token_id=tokenizer.pad_token_id
    )
    core = core_cls(
        model_cfg=cfg["model"],
        checkpointing=checkpointing
    )
    head = head_cls(
        model_cfg=cfg["model"],
        embedder=embedder if cfg["model"]["head"].get("tie_weights", False) else None
    )

    model = STLM(embedder, core, head)
    
    # print the size of the model
    total_params, trainable_params = count_parameters(model)
    print(f"Total params (excluding ties): {total_params:,}")
    print(f"Trainable params (excluding ties): {trainable_params:,}")
    
    return model


__all__ = [
    "BaseTokenizer",
    "BaseEmbedder",
    "BaseCore",
    "BaseHead",
    "STLM",
    "BaseTrainer",
    "SFTTrainer",
    "REGISTRY",
    "register_component",
    "build_from_config",
    "DDPWrapper",
    "FSDPWrapper",
    "LoRAWrapper",
]
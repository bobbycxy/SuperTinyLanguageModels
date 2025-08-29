import pkgutil
import importlib
import pathlib

from .core import STLM, BaseEmbedder, BaseCore, BaseHead, BaseTrainer, BaseTokenizer
from .registry import REGISTRY, register_component
from .tokenizers import build_tokenizer
from .wrappers import DDPWrapper, FSDPWrapper, LoRAWrapper
from .trainers.sft_trainer import SFTTrainer

# --- Auto-import all submodules so @register_component decorators run ---
package_dir = pathlib.Path(__file__).resolve().parent
for module_info in pkgutil.walk_packages([str(package_dir)], prefix="stlm."):
    # Skip __init__.py itself to avoid recursion
    if module_info.name != __name__:
        importlib.import_module(module_info.name)

def build_from_config(cfg: dict):
    """Build an STLM model (optionally with wrappers) from config."""
    print(REGISTRY["core"])
    print(cfg["model"]["core"]["name"])
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
        model_cfg=cfg["model"]["head"],
    )

    model = STLM(embedder, core, head)
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
#stlm/models/__init__.py

import importlib
import pkgutil
import pathlib

from stlm.registry import REGISTRY
from stlm.core import STLM
from stlm.tokenizers import build_tokenizer
from stlm.utils import count_parameters


def _import_all_model_submodules():
    """
    Recursively import all Python submodules under stlm/models so that
    components are registered via @register_component decorators.
    """
    package_dir = pathlib.Path(__file__).resolve().parent  # stlm/models
    prefix_base = "stlm.models"

    # Collect all directories under stlm/models
    subdirs = [str(package_dir)] + [
        str(p) for p in package_dir.rglob("*") if p.is_dir()
    ]

    imported = []
    for d in subdirs:
        # Build import prefix like 'stlm.models.embedders.' or 'stlm.models.cores.components.'
        rel_path = pathlib.Path(d).relative_to(package_dir)
        prefix = (
            f"{prefix_base}." + ".".join(rel_path.parts) + "."
            if rel_path.parts else f"{prefix_base}."
        )

        # Import all Python modules found within this directory
        for module_info in pkgutil.walk_packages([d], prefix=prefix):
            try:
                importlib.import_module(module_info.name)
                imported.append(module_info.name)
            except Exception as e:
                print(f"[WARN] Skipping module {module_info.name}: {e}")

    print(f"[stlm.models] Imported {len(imported)} model submodules.")
    return imported


# Automatically run this once on import
_import_all_model_submodules()

def build_from_config(cfg: dict):
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
    embedder = embedder_cls(
        model_cfg=model_cfg,
        checkpointing=checkpointing,
        pad_token_id=tokenizer.pad_token_id,
    )
    core = core_cls(
        model_cfg=model_cfg,
        checkpointing=checkpointing,
    )
    head = head_cls(
        model_cfg=model_cfg,
        embedder=embedder if model_cfg["head"].get("tie_weights", False) else None,
    )

    # Combine into the full model
    model = STLM(embedder, core, head, tokenizer)

    # Print parameter summary
    total_params, trainable_params = count_parameters(model)
    print(f"[Model] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model

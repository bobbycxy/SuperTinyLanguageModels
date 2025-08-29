# stlm/utils/hf_loader.py

import os
import torch
from transformers import AutoModelForCausalLM

_HF_MODEL_CACHE = {}

def get_hf_model(model_cfg):
    model_name = model_cfg["model_name"]

    if model_name in _HF_MODEL_CACHE:
        model = _HF_MODEL_CACHE[model_name]
    else:
        flash_attention = model_cfg.get("flash_attention", False)
        attn_implementation = "flash_attention_2" if flash_attention else "eager"

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            attn_implementation=attn_implementation,
            token=os.getenv("HF_ACCESS_TOKEN"),
            low_cpu_mem_usage=True,
            torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "float32"))
        )

        for p in model.parameters():
            p.requires_grad = False

        _HF_MODEL_CACHE[model_name] = model

    return model
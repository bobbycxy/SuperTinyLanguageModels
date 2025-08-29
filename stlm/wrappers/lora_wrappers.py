import torch.nn as nn
from stlm.utils.lora import apply_lora_to_model, count_lora_parameters

class LoRAWrapper(nn.Module):
    def __init__(self, model, lora_cfg: dict):
        super().__init__()
        self.model = apply_lora_to_model(
            model,
            target_modules=lora_cfg.get("target_modules"),
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32.0),
            dropout=lora_cfg.get("dropout", 0.0),
            exclude_modules=lora_cfg.get("exclude_modules")
        )
        lora_params, total_params, pct = count_lora_parameters(self.model)
        # print(f"[LoRAWrapper] Params: {lora_params:,} / {total_params:,} ({pct:.2f}%)")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

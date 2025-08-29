# stlm/cores/hf_transformer.py

from stlm import BaseCore
from stlm.utils import get_hf_model
from stlm.registry import register_component

@register_component("core", "huggingface")
class HuggingFaceCore(BaseCore):
    def __init__(self, model_cfg, checkpointing=False):
        super().__init__()
        model = get_hf_model(model_cfg=model_cfg)
        self.core = model.model
        self.checkpointing = checkpointing

        if self.checkpointing and hasattr(self.core, "gradient_checkpointing_enable"):
            self.core.gradient_checkpointing_enable()

    def forward(self, x, attn_mask=None):
        h = self.core(
            inputs_embeds=x,
            attention_mask=attn_mask,
            output_hidden_states=True
        ).hidden_states
        if isinstance(h, tuple):
            return h[-1]
        return h

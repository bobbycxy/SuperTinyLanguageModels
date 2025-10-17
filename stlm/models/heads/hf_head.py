# stlm/embedders/hf_head.py

from stlm import BaseHead
from stlm.utils import get_hf_model
from stlm.registry import register_component

@register_component("head", "huggingface")
class HuggingFaceHead(BaseHead):
    def __init__(self, model_cfg):
        super().__init__()
        model = get_hf_model(model_cfg=model_cfg)
        self.head = model.get_output_embeddings()

    def forward(self, x):
        return self.head(x)
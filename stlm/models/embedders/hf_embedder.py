# stlm/embedders/hf_embedder.py

from stlm import BaseEmbedder
from stlm.utils import get_hf_model
from stlm.registry import register_component

@register_component("embedder", "huggingface")
class HuggingFaceEmbedder(BaseEmbedder):
    def __init__(self, model_cfg, pad_token_id = 0, checkpointing=False):
        super().__init__()
        model = get_hf_model(model_cfg=model_cfg)
        self.embedding = model.get_input_embeddings()
        self.checkpointing = checkpointing

    def forward(self, token_ids):
        out = self.embedding(token_ids)
        # Ensure gradient flow if checkpointing is enabled
        if self.checkpointing and not out.requires_grad:
            out.requires_grad_(True)
        return out

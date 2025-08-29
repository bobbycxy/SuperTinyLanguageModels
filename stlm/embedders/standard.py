# stlm/embedders/hf_embedder.py

from stlm import BaseEmbedder
import torch
import torch.nn as nn
from stlm.registry import register_component

@register_component("embedder", "standard")
class StandardEmbedder(BaseEmbedder):
    def __init__(self, model_cfg, pad_token_id = 0, checkpointing=False):
        super().__init__()
        vocab_size = model_cfg["tokenizer"]["vocab_size"]
        max_position_embeddings = model_cfg["embedder"]["max_position_embeddings"]
        dropout_rate = model_cfg["embedder"]["dropout"]
        
        self.hidden_size = model_cfg["embedder"]["hidden_size"]
        self.checkpointing = checkpointing
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=pad_token_id
        )
        self.pos_embedding = nn.Embedding( # This can probably be another class to try out different class
            num_embeddings=max_position_embeddings,
            embedding_dim=self.hidden_size
        )
        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        B, T = token_ids.size()
        device = token_ids.device

        token_embs = self.token_embedding(token_ids) # [B, T, H]
        pos_ids = torch.arange(end=T, device=device).unsqueeze(0).expand(B, -1)
        pos_embs = self.pos_embedding(pos_ids) # [B, T, H]
        embeddings = (token_embs + pos_embs) * (self.hidden_size ** 0.5)
        embeddings = self.dropout(embeddings)

        # Ensure gradient flow if checkpointing is enabled
        if self.checkpointing and not embeddings.requires_grad:
            embeddings.requires_grad_(True)
        
        return embeddings # [B, T, H]

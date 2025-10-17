from .hf_tokenizer import HuggingFaceTokenizer
from .bpe import ByteBPETokenizer

def build_tokenizer(cfg):
    tok_cfg = cfg["model"]["tokenizer"]
    if tok_cfg["type"] == "huggingface":
        return HuggingFaceTokenizer(tok_cfg["model_name"])
    elif tok_cfg["type"] == "bytebpe":
        return ByteBPETokenizer.from_config(cfg)
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_cfg["type"]}")
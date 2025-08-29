# from stlm.tokenizers.bpe import ByteBPETokenizer

# texts = ["hello world", "LLMs are fun", "hello mate this is a helluva world we are in"]
# tok = ByteBPETokenizer(vocab_size=500)
# tok.train(texts)
# tok.save("byte_bpe.json")

# tok2 = ByteBPETokenizer.load("byte_bpe.json")
# enc = tok2.batch_encode(["hello world", "hello in"])
# print(enc["input_ids"])
# print(tok2.batch_decode(enc["input_ids"].tolist()))

from stlm.tokenizers import build_tokenizer
from stlm.tokenizers.bpe import ByteBPETokenizer
import yaml

with open("stlm/configs/sft.yaml", "r") as f:
    cfg = yaml.safe_load(f)

tokenizer = build_tokenizer(cfg)
tok2 = ByteBPETokenizer.load("byte_bpe.json")
enc = tok2.batch_encode(["hello world", "hello in"])
print(enc["input_ids"])
print(tok2.batch_decode(enc["input_ids"]))
import os
import torch
from tqdm import tqdm
from heapq import nlargest
from collections import Counter
from datasets import load_dataset
from stlm.core import BaseTokenizer
import torch.distributed as dist

class ByteBPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size=5000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        # Special tokens
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.special_tokens = [self.pad_token, self.eos_token]
        self.pad_token_id = 0
        self.eos_token_id = 1

        self.vocab = {}
        self.inv_vocab = {}
        self.merges = {}  # (int,int) → new_id
        self.byte_vocab = {i: bytes([i]) for i in range(256)}

    # ---------------- Save / Load ----------------
    def save(self, path: str):
        import json
        obj = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "merges": list(self.merges.items()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        tok = cls(vocab_size=obj["vocab_size"])
        tok.special_tokens = obj["special_tokens"]
        tok.pad_token_id = obj["pad_token_id"]
        tok.eos_token_id = obj["eos_token_id"]
        tok.merges = {tuple(map(int, k)): v for k, v in obj["merges"]}
        tok._build_vocab()
        return tok

    # ---------------- Training ----------------
    def train(self, texts, verbose=True, max_clutch_size=64):
        # Flatten dataset to raw bytes
        if isinstance(texts, list):
            dataset_text = " ".join(texts)  # join all text
        else:
            dataset_text = str(texts)

        byte_seq = list(dataset_text.encode("utf-8"))
        ids = list(byte_seq)

        current_vocab_size = 256
        num_merges = self.vocab_size - current_vocab_size - len(self.special_tokens)
        merges = {}

        with tqdm(total=num_merges, desc="Training BPE", disable=not verbose) as pbar:
            while num_merges > 0:
                stats = self._get_stats(ids)
                top_pairs = nlargest(min(max_clutch_size, num_merges), stats, key=stats.get)

                pairs_to_merge = {}
                first_seen, second_seen = set(), set()
                for pair in top_pairs:
                    if pair[0] in second_seen or pair[1] in first_seen:
                        first_seen.add(pair[0])
                        second_seen.add(pair[1])
                        continue
                    first_seen.add(pair[0])
                    second_seen.add(pair[1])
                    pairs_to_merge[pair] = current_vocab_size
                    current_vocab_size += 1
                    num_merges -= 1
                    pbar.update(1)

                ids = self._multi_merge(ids, pairs_to_merge)
                merges.update(pairs_to_merge)

        self.merges = merges
        self._build_vocab()

    def _get_stats(self, ids):
        stats = Counter()
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            stats[pair] += 1
        return stats

    def _multi_merge(self, ids, pairs_to_merge):
        i = 0
        new_ids = []
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) in pairs_to_merge:
                new_ids.append(pairs_to_merge[(ids[i], ids[i + 1])])
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _build_vocab(self):
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        vocab[self.pad_token_id] = self.pad_token.encode("utf-8")
        vocab[self.eos_token_id] = self.eos_token.encode("utf-8")
        self.vocab = vocab
        self.inv_vocab = {i: tok for i, tok in vocab.items()}
    
    @classmethod
    def from_config(cls, cfg):
        tok_cfg = cfg["model"]["tokenizer"]
        ds_cfg = cfg["trainer"]["dataset"]
        save_path = tok_cfg.get("save_path", "byte_bpe.json")

        if os.path.exists(save_path):
            print(f"✅ Loading ByteBPE tokenizer from {save_path}")
            return cls.load(save_path)
        
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

        if rank == 0:
            print("ℹ️ Training new ByteBPE tokenizer...")
            dataset = load_dataset(ds_cfg["path"], name=ds_cfg.get("name", None), split="train")
            texts = dataset[ds_cfg.get("text_column", "text")]

            tokenizer = cls(vocab_size=tok_cfg.get("vocab_size", 5000))
            tokenizer.train(texts)
            tokenizer.save(save_path)
            print(f"✅ Saved ByteBPE tokenizer to {save_path}")
        
        dist.barrier() if (dist.is_available() and dist.is_initialized()) else None
        
        return cls.load(save_path)

    # ---------------- Encode / Decode ----------------
    def encode(self, text: str, add_eos=True, max_length=None):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self._multi_merge(ids, {pair: idx})
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length:
            ids = ids[-max_length:]
        return ids

    def batch_encode(self, texts, add_eos=True, max_length=None):
        # Encode each text individually
        encoded = [self.encode(t, add_eos, max_length) for t in texts]
        # Use existing pad_batch to return tensors
        return self.pad_batch(encoded)


    def decode(self, token_ids):
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        text_bytes = b"".join(self.vocab[idx] for idx in token_ids if idx in self.vocab)
        return text_bytes.decode("utf-8", errors="replace")

    def batch_decode(self, token_lists):
        if isinstance(token_lists, dict):  # accept dict from batch_encode
            token_lists = token_lists["input_ids"]
        return [self.decode(seq) for seq in token_lists]

    def pad_batch(self, batch_token_ids, direction="right"):
        max_len = max((len(x) for x in batch_token_ids), default=0)
        padded, masks = [], []
        for seq in batch_token_ids:
            pad_len = max_len - len(seq)
            if direction == "right":
                padded.append(seq + [self.pad_token_id] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
            else:
                padded.append([self.pad_token_id] * pad_len + seq)
                masks.append([0] * pad_len + [1] * len(seq))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

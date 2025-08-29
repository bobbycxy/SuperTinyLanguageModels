from transformers import AutoTokenizer
from typing import List, Dict, Any
import torch
from stlm import BaseTokenizer

class HuggingFaceTokenizer(BaseTokenizer):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str, add_eos: bool = True, max_length: int = None) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            ids.append(self.eos_token_id)
        if max_length:
            ids = ids[-max_length:]
        return ids
    
    def batch_encode(self, texts: List[str], add_eos: bool = True, max_length: int = None) -> Dict[str, Any]:
        out = self.tokenizer(texts, truncation=True, padding=False, max_length=max_length, add_special_tokens=False)
        if add_eos:
            out["input_ids"] = [ids + [self.eos_token_id] for ids in out["input_ids"]]
            out["attention_mask"] = [mask + [1] for mask in out["attention_mask"]]
        return self.pad_batch(out["input_ids"])
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def batch_decode(self, batch_token_ids: List[List[int]]) -> List[str]:
        return self.tokenizer.batch_decode(batch_token_ids, skip_special_tokens=True)

    def pad_batch(self, batch_token_ids: List[List[int]], direction: str = "right") -> Dict[str, Any]:
        max_len = max(len(x) for x in batch_token_ids)
        padded = []
        masks = []
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
            "attention_mask": torch.tensor(masks, dtype=torch.long)
        }
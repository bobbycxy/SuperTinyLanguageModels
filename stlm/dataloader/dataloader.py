# stlm/dataloader/dataloader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from stlm.utils.data_utils import get_file_path


def prepare_data(cfg, tokenizer):
    import torch.distributed as dist

    # Detect distributed setup
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    out_dir = os.path.join(get_file_path(cfg), "tokenized")
    os.makedirs(out_dir, exist_ok=True)

    # Skip if data already exists
    if rank == 0:
        if all(os.path.exists(os.path.join(out_dir, f"{s}.bin")) for s in ["train", "validation"]):
            print(f"Tokenized data already exists at {out_dir}. Skipping preprocessing.")
        else:
            ds_cfg = cfg["trainer"]["dataset"]
            dataset = load_dataset(ds_cfg["path"], ds_cfg.get("name", "20231101.simple"), split="train")

            # Auto train/validation split
            split_ratio = ds_cfg.get("val_split", 0.01)
            print(f"Splitting train into train/validation ({split_ratio})")
            split_dict = dataset.train_test_split(test_size=split_ratio, seed=42)
            datasets = {"train": split_dict["train"], "validation": split_dict["test"]}

            text_col = ds_cfg.get("text_column", "text")
            for split, dset in datasets.items():
                print(f"[Rank 0] 🔹 Tokenizing {split}...")
                tokenized = dset.map(
                    lambda b: {"input_ids": [tokenizer.encode(t, add_eos=True) for t in b[text_col]]},
                    batched=True,
                    remove_columns=dset.column_names,
                    num_proc=min(32, os.cpu_count()),
                )

                all_ids = np.concatenate([np.array(x, dtype=np.uint16) for x in tokenized["input_ids"]])
                filename = os.path.join(out_dir, f"{split}.bin")

                arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(len(all_ids),))
                arr[:] = all_ids
                arr.flush()

                print(f"[Rank 0] Wrote {split}.bin ({len(all_ids):,} tokens, dtype=uint16)")

    # Barrier ensures all ranks wait for rank 0 to finish
    if is_distributed:
        dist.barrier()
        if rank != 0:
            print(f"[Rank {rank}] ⏳ Data ready, proceeding with training.")


class TokenizedDataset(Dataset):
    """
    Unified dataset for train and validation.
    Uses map-style indexing for compatibility with DistributedSampler.
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.context_window = cfg["model"]["embedder"]["max_position_embeddings"]

        data_dir = os.path.join(get_file_path(cfg), "tokenized")
        self.data_path = os.path.join(data_dir, f"{split}.bin")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Tokenized data file not found: {self.data_path}")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.data_len = len(self.data)
        self.max_start = self.data_len - self.context_window

        # For validation: non-overlapping windows
        if split == "validation":
            self.indices = np.arange(0, self.max_start, self.context_window, dtype=np.int64)
        else:
            self.indices = np.arange(self.max_start, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = torch.from_numpy(self.data[i : i + self.context_window].astype(np.int64))
        y = torch.from_numpy(self.data[i + 1 : i + 1 + self.context_window].astype(np.int64))
        return {"input_ids": x, "labels": y}


def get_dataloaders(cfg, split="train"):
    dataset = TokenizedDataset(cfg, split=split)
    is_train = split == "train"
    is_distributed = torch.distributed.is_initialized()

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=is_train,
            drop_last=is_train,
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=cfg["trainer"].get("num_workers", 8 if is_train else 0),
        pin_memory=True,
        persistent_workers=is_train, 
        drop_last=is_train,
    )

    return loader

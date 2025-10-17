from stlm.utils.data_utils import get_file_path
from datasets import load_dataset, load_from_disk
import os
from datasets import DatasetDict

def archive_prepare_tokenized_dataset(cfg, tokenizer):
    out_dir = os.path.join(get_file_path(cfg), "tokenized")
    if os.path.exists(out_dir):
        print(f"✅ Tokenized dataset already at {out_dir}")
        return

    ds_cfg = cfg["trainer"]["dataset"]

    # Always build a DatasetDict from your list of splits
    datasets = {
        split: load_dataset(
            path=ds_cfg["path"],
            name=ds_cfg.get("name", None),
            split=split
        )
        for split in ds_cfg["splits"]
    }
    dataset = DatasetDict(datasets)

    if "train" in dataset and "validation" not in dataset:
        split_ratio = ds_cfg.get("val_split", 0.01)  # default 1%
        print(f"ℹ️ Splitting train into train/validation with ratio {split_ratio}")
        split_dict = dataset["train"].train_test_split(test_size=split_ratio, seed=42)
        dataset["train"] = split_dict["train"]
        dataset["validation"] = split_dict["test"]

    text_col = ds_cfg.get("text_column", "text")

    def tokenize_fn(batch):
        return tokenizer.batch_encode(
            batch[text_col],
            add_eos=True,
            max_length=cfg["model"]["embedder"]["max_position_embeddings"],
        )

    # Apply map per split
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=min(32, os.cpu_count())
    )

    tokenized.save_to_disk(out_dir)
    print(f"✅ Saved tokenized dataset to {out_dir}")

def load_tokenized_dataset(cfg):
    return load_from_disk(os.path.join(get_file_path(cfg), "tokenized"))
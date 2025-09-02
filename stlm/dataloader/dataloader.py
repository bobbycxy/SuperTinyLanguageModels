from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch
from stlm.dataloader.utils import prepare_tokenized_dataset, load_tokenized_dataset
import torch.distributed as dist


def collate_batch(batch, pad_token_id, ignore_index: int = -100):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attn_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index

    return {
        "input_ids": input_ids,
        "attention_mask": attn_masks,
        "labels": labels
    }


def get_dataloaders(cfg, tokenizer):
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if rank == 0:
        prepare_tokenized_dataset(cfg=cfg, tokenizer=tokenizer)
    
    # make all ranks wait till the dataset is ready
    if is_distributed:
        dist.barrier()

    dataset = load_tokenized_dataset(cfg=cfg)
    world_size = dist.get_world_size() if is_distributed else 1
    global_batch = cfg["trainer"]["batch_size"]  # now treated as GLOBAL batch
    grad_accum = cfg["trainer"].get("grad_accum_steps", 1)

    assert global_batch % (world_size * grad_accum) == 0, \
        f"Global batch {global_batch} must divide evenly into world_size×grad_accum."

    per_device_batch = global_batch // (world_size * grad_accum)

    def make_loader(split, shuffle):
        if split not in dataset:
            return None
        sampler = DistributedSampler(dataset[split]) if is_distributed else None
        return DataLoader(
            dataset[split],
            batch_size=per_device_batch,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
        )

    train_dl = make_loader("train", shuffle=True)
    val_dl = make_loader("validation", shuffle=False)

    return train_dl, val_dl
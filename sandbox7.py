import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
import torch.distributed as dist

from stlm.utils.lora import get_lora_parameters, count_lora_parameters, apply_lora_to_model
from stlm.trainers.sft_trainer import SFTTrainer, reduce_mean
from stlm.tokenizers import build_tokenizer
from stlm.data.dataloader import get_dataloaders

from tqdm import tqdm
import wandb
from stlm.utils import count_parameters

def main():
    # ------------------ Init ------------------
    init_distributed_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YAML config first
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # only rank 0 initializes wandb
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="supertinylanguagemodels",
            config={**cfg, "device": device},  # merge yaml + device
        )

    tokenizer = build_tokenizer(cfg=cfg)
    train_dataloader, val_dataloader = get_dataloaders(cfg, tokenizer=tokenizer)

    model = stlm.build_from_config(cfg)

    total_params, trainable_params = count_parameters(model, verbose=True)
    print(f"Total params (excluding ties): {total_params:,}")
    print(f"Trainable params (excluding ties): {trainable_params:,}")

    model = stlm.LoRAWrapper(model, lora_cfg=cfg["lora"])
    model = stlm.FSDPWrapper(model, device=device)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=float(cfg["trainer"].get("lr", 1e-3))
    )

    trainer = SFTTrainer(
        model,
        optimizer,
        scheduler=None,
        device=device,
        grad_accum_steps=cfg["trainer"].get("grad_accum_steps", 1)
    )

    for epoch in range(cfg["trainer"].get("epochs", 1)):
        accum_count = 0
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            metrics = trainer.train_step(batch)
            accum_count += 1

            if accum_count % trainer.grad_accum_steps == 0:
                trainer.optimizer_step(step=step, epoch=epoch)

            if accum_count % cfg["trainer"]["eval_interval"] == 0:
                val_losses = []
                for batch in val_dataloader:   # every rank runs this
                    val_loss = trainer.eval_step(batch)
                    val_losses.append(val_loss)

                # mean across batches
                mean_val_loss = sum(val_losses) / len(val_losses)

                # only rank 0 logs
                if not dist.is_initialized() or dist.get_rank() == 0:
                    wandb.log({"val/loss": mean_val_loss}, step=trainer.processed_tokens)
                    print(f"[VAL] epoch={epoch} loss={mean_val_loss:.4f}")

        if accum_count % trainer.grad_accum_steps != 0:
            trainer.optimizer_step()

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
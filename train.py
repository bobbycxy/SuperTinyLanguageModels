import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
import torch.distributed as dist

from stlm.utils.lora import get_lora_parameters, count_lora_parameters, apply_lora_to_model
from stlm.trainers.causaltrainer import CausalTrainer
from stlm.tokenizers import build_tokenizer
from stlm.dataloader.dataloader import get_dataloaders, prepare_data

from tqdm import tqdm
import wandb
from stlm.utils import count_parameters

def main():
    # ------------------ Init ------------------
    rank, world_size, device_id = init_distributed_setup()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Load YAML config first
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # only rank 0 initializes wandb
    if cfg["general"]["wandb"].get("wandb_log", False) and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(
            project=cfg["general"]["wandb"].get("project_name", "supertinylanguagemodels"),
            config={**cfg, "device": str(device)}
        )

    # tokenizer and dataloaders
    tokenizer = build_tokenizer(cfg=cfg)
    prepare_data(cfg, tokenizer=tokenizer)
    train_dataloader = get_dataloaders(cfg, split="train")
    val_dataloader = get_dataloaders(cfg, split="validation")

    # load build the model
    model = stlm.build_from_config(cfg)

    # Distributed wrapper
    model = stlm.DDPWrapper(model, device_id=device_id)

    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=float(cfg["trainer"].get("lr", 1e-3))
    )

    # Trainer
    trainer = CausalTrainer(
        model,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=None,
        device=device,
        grad_accum_steps=cfg["trainer"].get("grad_accum_steps", 1)
    )

    # Training Loop
    trainer.eval_step(step=0, prompt="Once upon a time", max_new_tokens=50)
    for step in range(cfg["trainer"]["max_iterations"]):
        trainer.train_step(step)
        if step % cfg["trainer"]["eval_interval"] == 0 and step != 0:
            trainer.eval_step(step, prompt="Once upon a time", max_new_tokens=50)
    
    # Cleanup
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
# train.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
import torch.distributed as dist
import wandb

import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
from stlm.trainers.causaltrainer import CausalTrainer
from stlm.tokenizers import build_tokenizer
from stlm.dataloader.dataloader import get_dataloaders, prepare_data
import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="stlm/configs", config_name="sft")
def main(cfg: DictConfig):
    # Hydra automatically creates and chdirs into outputs/<timestamp>/
    run_dir = os.getcwd()
    rank, world_size, device_id = init_distributed_setup()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ------------------ W&B ------------------
    if cfg.general.wandb.get("wandb_log", False) and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.init(
            project=cfg.general.wandb.get("project_name", "stlm"),
            name=cfg.general.get("run_name", os.path.basename(run_dir)),
            dir=run_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # ------------------ Model + Data ------------------
    tokenizer = build_tokenizer(cfg)
    prepare_data(cfg, tokenizer)
    train_dataloader = get_dataloaders(cfg, split="train")
    val_dataloader = get_dataloaders(cfg, split="validation")

    model = stlm.build_from_config(cfg)

    # --- Load checkpoint before wrapping ---
    ckpt_path = cfg.general.get("load_checkpoint", None)
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        log.info(f"Preloaded checkpoint from {ckpt_path}")
    else:
        log.info(f"No checkpoint found at {ckpt_path}, training from scratch.")

    model = stlm.DDPWrapper(model, device_id=device_id)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.trainer.get("lr", 1e-3),
    )

    trainer = CausalTrainer(
        model,
        optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=None,
        device=device,
        grad_accum_steps=cfg.trainer.get("grad_accum_steps", 1),
        out_dir=cfg.general.get("outputs_dir"),
    )

    # val
    trainer.eval_step(
        step=0,
        prompt=cfg.general.get("eval_prompt", "Once upon a time"),
        max_new_tokens=50,
    )

    # ------------------ Train loop ------------------
    for step in range(1, cfg.trainer.get("max_iterations", 1000) + 1):
        trainer.train_step(step)
        if step % cfg.trainer.get("eval_interval", 100) == 0:
            trainer.eval_step(step, prompt=cfg.general.get("eval_prompt", "Once upon a time"), max_new_tokens=50)
        if step % cfg.trainer.get("save_interval", 1000) == 0:
            trainer.save_checkpoint(step)

    if is_dist():
        dist.destroy_process_group()
    if cfg.general.wandb.get("wandb_log", False) and (not dist.is_initialized() or dist.get_rank() == 0):
        wandb.finish()

if __name__ == "__main__":
    main()

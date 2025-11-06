import os
import torch
import matplotlib.pyplot as plt
import yaml
from omegaconf import OmegaConf
from stlm.schedulers.lr_schedulers import build_lr_scheduler

# --- Load config safely ---
cfg = OmegaConf.load("stlm/configs/generic.yaml")
if "general" in cfg:
    del cfg["general"]      # drop Hydra-only keys
cfg = OmegaConf.to_container(cfg, resolve=True)

# --- Create model and optimizer ---
model = torch.nn.Linear(10, 10)
base_lr = float(cfg["trainer"]["lr"])
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

# --- Build scheduler and plot ---
scheduler = build_lr_scheduler(optimizer, cfg)
# --- Track learning rate changes ---
total_steps = cfg["trainer"]["max_iterations"]
lrs = []

for step in range(total_steps):
    optimizer.step()
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    lrs.append(lr)

# --- Plot learning rate schedule ---
plt.figure(figsize=(8, 4))
plt.plot(range(total_steps), lrs, label=cfg["trainer"]["scheduler"]["name"])
plt.title("Learning Rate Schedule from sft.yaml")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- Save plot ---
save_path = "outputs/lr_schedule_sft.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
print(f"✅ Saved LR schedule plot to {os.path.abspath(save_path)}")

plt.show()

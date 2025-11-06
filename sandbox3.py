# train.py
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

import stlm
import logging
log = logging.getLogger(__name__)

@hydra.main(config_path="stlm/configs", config_name="sft")
def main(cfg: DictConfig):

    # ------------------ Model + Data ------------------
    model = stlm.STLM.from_config(cfg)

    # --- Load checkpoint before wrapping ---
    ckpt_path = "/home/bobby/code-repo/astar-projects/SuperTinyLanguageModels/outputs/2025-10-18/22-46-49/checkpoints/checkpoint_step_10000.pt"
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        log.info(f"Preloaded checkpoint from {ckpt_path}")
    else:
        log.info(f"No checkpoint found at {ckpt_path}, training from scratch.")

    prompts = ["What is the capital of France?","The sun rises in the"]

    for prompt in prompts:
        res = model.generate(
            prompt,
            max_new_tokens=50,
            eos_token_id=model.tokenizer.eos_token_id,
        )
        print(f"Prompt: {prompt}\nGenerated: {res}\n")

if __name__ == "__main__":
    main()

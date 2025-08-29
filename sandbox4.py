import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
import torch.distributed as dist

# import the LoRA helpers we added
from stlm.utils.lora import get_lora_parameters, count_lora_parameters

def main():
    # init dist env
    init_distributed_setup()

    # Load YAML config
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Build model
    model = stlm.build_from_config(cfg)
    model = stlm.LoRAWrapper(model, cfg["lora"])
    model = stlm.FSDPWrapper(model, device="cuda")

    # Print param stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params, total_params2, pct = count_lora_parameters(model)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA Parameters: {lora_params:,} ({pct:.2f}% of total)")

    # Dummy dataset: vocab size = from config (or default 50257)
    vocab_size = cfg.get("model", {}).get("head", {}).get("vocab_size", 50257)
    batch_size, seq_len = 2, 16
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer: only LoRA params
    optimizer = optim.AdamW(get_lora_parameters(model), lr=1e-3)

    # Forward pass
    logits = model(dummy_input)  # [B, T, V]
    loss = criterion(logits.view(-1, logits.size(-1)), dummy_labels.view(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Dummy training step complete. Loss = {loss.item():.4f}")

    if is_dist():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
import torch.distributed as dist

from stlm.utils.lora import get_lora_parameters, count_lora_parameters, apply_lora_to_model
from stlm.trainers.sft_trainer import SFTTrainer


def main():
    # ------------------ Init ------------------
    init_distributed_setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YAML config
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------ Build Model ------------------
    # 1. Build base model
    model = stlm.build_from_config(cfg)

    # 2. Apply LoRA
    model = stlm.LoRAWrapper(model, lora_cfg=cfg["lora"])

    # 4. Wrap with FSDP
    model = stlm.FSDPWrapper(model, device=device)

    # Verify we have trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found! Check LoRA initialization.")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Trainable parameter ratio: {trainable_count/total_params*100:.2f}%")

    # Debug: Check if any parameters require grad
    if dist.get_rank() == 0:
        requires_grad_count = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"[Rank 0] Parameters requiring grad: {requires_grad_count}")
        
        # Sample some parameter names that require grad
        grad_params = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad][:5]
        print(f"[Rank 0] Sample trainable params: {grad_params}")

    # ------------------ Optimizer ------------------
    # Use only trainable parameters
    optimizer = optim.AdamW(trainable_params, lr=float(cfg["trainer"].get("lr", 1e-3)))

    trainer = SFTTrainer(
        model,
        optimizer,
        scheduler=None,
        device=device,
        grad_accum_steps=cfg["trainer"].get("grad_accum_steps", 1)
    )

    # ------------------ Dummy Data ------------------
    vocab_size, seq_len = 50257, 128
    train_dataloader = []
    for _ in range(20):  # fewer steps for sandbox
        input_ids = torch.randint(0, vocab_size, (cfg["trainer"]["batch_size"], seq_len), device=device)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # shift left
        labels[:, -1] = -100               # ignore last token
        attn_mask = torch.ones(cfg["trainer"]["batch_size"], seq_len, device=device)
        train_dataloader.append((input_ids, labels, attn_mask))

    # ------------------ Train Loop ------------------
    for epoch in range(cfg["trainer"].get("epochs", 1)):
        accum_count = 0
        for step, batch in enumerate(train_dataloader):
            try:
                metrics = trainer.train_step(batch)
                accum_count += 1

                if accum_count % trainer.grad_accum_steps == 0:
                    trainer.optimizer_step()

                if step % cfg["trainer"].get("log_interval", 10) == 0:
                    trainer.log_metrics(metrics, prefix="Train", step=step, epoch=epoch)
                    print(f"[Mem] Allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB | "
                          f"Reserved: {torch.cuda.memory_reserved()/1e6:.1f} MB")
                    
            except RuntimeError as e:
                if "does not require grad" in str(e):
                    print(f"[ERROR] No gradients available at step {step}")
                    print("Checking model parameters...")
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            print(f"  {name}: requires_grad=True, grad={param.grad is not None}")
                            break
                    else:
                        print("  No parameters require gradients!")
                raise e

        if accum_count % trainer.grad_accum_steps != 0:
            trainer.optimizer_step()

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
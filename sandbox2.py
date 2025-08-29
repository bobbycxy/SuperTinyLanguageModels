import torch
import yaml
import stlm
from stlm.utils.dist_utils import init_distributed_setup, is_dist
import torch.distributed as dist

def main():
    # init dist env
    init_distributed_setup()

    # Load YAML config
    with open("stlm/configs/sft.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Build model directly (wrappers handled inside)
    model = stlm.build_from_config(cfg)
    # model = stlm.FSDPWrapper(model, device="cuda")
    model.core = stlm.CheckpointWrapper(model.core)    


    # Total params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    if is_dist():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

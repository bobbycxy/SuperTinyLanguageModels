import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from transformers import AutoModelForCausalLM, AutoTokenizer

# import your LoRA
from stlm.utils.lora import apply_lora_to_model
from functools import partial
from stlm.utils import get_transformer_layer_classes


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # ------------------ 1. Load HuggingFace model ------------------
    model_name = "facebook/opt-350m"  # demo model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # ------------------ 2. Apply LoRA (your implementation) ------------------
    model = apply_lora_to_model(
        model,
        target_modules=["q_proj", "v_proj"],  # OPT/LLaMA style
        rank=8,
        alpha=16,
        dropout=0.05,
        exclude_modules=[]
    )

    dtype = next(model.parameters()).dtype
    for p in model.parameters():
        if p.dtype != dtype:
            p.data = p.data.to(dtype)

    # ------------------ 3. Enable gradient checkpointing ------------------
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print("[checkpoint] enabled")

    # ------------------ 4. Wrap with FSDP ------------------
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # Figure out transformer block classes
    transformer_layer_classes = get_transformer_layer_classes(model)

    # Wrap policy
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layer_classes)

    # Now FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        use_orig_params=True,
    )

    # ------------------ 5. Training setup ------------------
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )
    scaler = ShardedGradScaler()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # dummy batch
    inputs = tokenizer("Hello world!", return_tensors="pt").to(device)

    # ------------------ 6. Training step ------------------
    model.train()
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        print("Train step finished, loss:", loss.item())

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

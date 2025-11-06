from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_lr_scheduler(optimizer, cfg):
    scheduler_name = cfg["trainer"]["scheduler"]["name"].lower()
    params = cfg["trainer"]["scheduler"].get("params", {})
    total_steps = cfg["trainer"].get("max_iterations", 1000)
    warmup_ratio = cfg["trainer"]["scheduler"].get("warmup_ratio", 0.1)
    decay_ratio = cfg["trainer"]["scheduler"].get("decay_ratio", 0.5)

    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = int(total_steps * decay_ratio)
    hold_steps = total_steps - warmup_steps - decay_steps

    # --- Warmup phase ---
    warmup_scheduler = (
        LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        ) if warmup_steps > 0 else None
    )

    # --- Main scheduler ---
    if scheduler_name == "linearlr":
        linear_params = dict(params)
        if "total_iters" not in linear_params:
            linear_params["total_iters"] = decay_steps
        main_scheduler = LinearLR(optimizer, **linear_params)
        final_lr_factor = linear_params.get("end_factor", 0.01)

    elif scheduler_name == "cosineannealinglr":
        # Inject T_max dynamically if not provided
        cosine_params = dict(params)
        if "T_max" not in cosine_params:
            cosine_params["T_max"] = max(decay_steps, 1)
        main_scheduler = CosineAnnealingLR(optimizer, **cosine_params)

        eta_min = cosine_params.get("eta_min", 0.0)
        final_lr_factor = eta_min / optimizer.defaults["lr"]

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # --- Hold phase (flat continuation) ---
    if hold_steps > 0:
        hold_scheduler = LinearLR(
            optimizer,
            start_factor=final_lr_factor,
            end_factor=final_lr_factor,
            total_iters=hold_steps
        )

        schedulers = [s for s in [warmup_scheduler, main_scheduler, hold_scheduler] if s is not None]
        milestones = []
        if warmup_scheduler:
            milestones.append(warmup_steps)
        milestones.append(warmup_steps + decay_steps)

        scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
    else:
        scheduler = main_scheduler if warmup_scheduler is None else SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )

    print(f"[Scheduler: {scheduler_name}] Warmup={warmup_steps}, Decay={decay_steps}, Hold={hold_steps}, T_max={decay_steps}")
    return scheduler

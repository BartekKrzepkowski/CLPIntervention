"""Controlled BatchNorm recalibration for phase-boundary diagnostics."""

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from src.utils.utils_trainer import capture_rng_state, restore_rng_state


def _selected_batchnorm(model, scope):
    analysis_model = model.main_model if hasattr(model, "main_model") else model
    if scope not in {"main_branch", "all"}:
        raise ValueError("BatchNorm recalibration scope must be main_branch or all")
    selected = []
    for name, module in analysis_model.named_modules():
        if not isinstance(module, _BatchNorm) or not module.track_running_stats:
            continue
        if scope == "all" or name == "main_branch" or name.startswith("main_branch."):
            selected.append((name, module))
    if not selected:
        raise ValueError(f"no tracked BatchNorm modules found for scope={scope}")
    return selected


def _snapshot(modules):
    return {
        name: {
            "running_mean": module.running_mean.detach().clone(),
            "running_var": module.running_var.detach().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().clone(),
            "momentum": module.momentum,
        }
        for name, module in modules
    }


def _restore_snapshot(modules, snapshot):
    for name, module in modules:
        state = snapshot[name]
        module.running_mean.copy_(state["running_mean"])
        module.running_var.copy_(state["running_var"])
        module.num_batches_tracked.copy_(state["num_batches_tracked"])


@torch.no_grad()
def recalibrate_batchnorm(
    model,
    loader,
    device,
    *,
    num_batches,
    scope="main_branch",
):
    """Recompute selected BN buffers without changing weights or consuming RNG."""
    num_batches = int(num_batches)
    if num_batches < 1:
        raise ValueError("num_batches must be positive")
    selected = _selected_batchnorm(model, scope)
    before = _snapshot(selected)
    module_modes = [(module, module.training) for module in model.modules()]
    rng_state = capture_rng_state()
    completed = 0
    try:
        model.eval()
        for _name, module in selected:
            module.reset_running_stats()
            module.momentum = None
            module.train()

        for batch in loader:
            (x_left, x_right), _targets = batch
            model(
                x_left.to(device),
                x_right.to(device),
                enable_left_branch=True,
                enable_right_branch=True,
            )
            completed += 1
            if completed >= num_batches:
                break
        if completed == 0:
            raise ValueError("cannot recalibrate BatchNorm with an empty loader")
    except Exception:
        _restore_snapshot(selected, before)
        raise
    finally:
        for name, module in selected:
            module.momentum = before[name]["momentum"]
        for module, was_training in module_modes:
            module.training = was_training
        restore_rng_state(rng_state)

    mean_delta_sq = 0.0
    var_delta_sq = 0.0
    features = 0
    for name, module in selected:
        mean_delta_sq += float(
            torch.sum((module.running_mean - before[name]["running_mean"]) ** 2)
        )
        var_delta_sq += float(
            torch.sum((module.running_var - before[name]["running_var"]) ** 2)
        )
        features += int(module.running_mean.numel())
    return {
        "bn_recalibration/batches": completed,
        "bn_recalibration/modules": len(selected),
        "bn_recalibration/features": features,
        "bn_recalibration/running_mean_delta_l2": mean_delta_sq ** 0.5,
        "bn_recalibration/running_var_delta_l2": var_delta_sq ** 0.5,
    }

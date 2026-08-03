"""Deterministic validation-probe gradient diagnostics for Phase 3."""

from __future__ import annotations

import math

import torch

from src.trainer.modality_evaluation import (
    DOMINANT_ONLY_MODE,
    FULL_MODE,
    WEAK_ONLY_MODE,
)


def _module_modes(module):
    if not isinstance(module, torch.nn.Module):
        return None
    return [(child, child.training) for child in module.modules()]


def _restore_module_modes(states):
    if states is None:
        return
    for module, training in states:
        module.training = training


def _loss(criterion, logits, targets):
    result = criterion(logits, targets)
    return result[0] if isinstance(result, tuple) else result


def _named_trainable_parameters(model, marker):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and marker in name
    ]


def _gradients(loss, parameters):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )


def _norm(gradients):
    squared = sum(
        gradient.detach().to(torch.float64).square().sum()
        for gradient in gradients
    )
    return float(torch.sqrt(squared))


def _cosine(left, right):
    dot = sum(
        (first.detach().to(torch.float64) * second.detach().to(torch.float64))
        .sum()
        for first, second in zip(left, right)
    )
    left_norm = torch.sqrt(
        sum(value.detach().to(torch.float64).square().sum() for value in left)
    )
    right_norm = torch.sqrt(
        sum(value.detach().to(torch.float64).square().sum() for value in right)
    )
    denominator = left_norm * right_norm
    if float(denominator) == 0.0:
        return float("nan")
    return float(dot / denominator)


def evaluate_phase3_gradient_diagnostics(
    model,
    criterion,
    loader,
    device,
    *,
    max_batches=1,
):
    """Measure validation gradients without changing parameters or .grad.

    The shared-trunk gradient is evaluated in full, dominant-only and
    weak-only modes. The weak-branch norm is evaluated in weak-only mode.
    BatchNorm remains in evaluation mode and previous modes are restored.
    """
    if int(max_batches) < 1:
        raise ValueError("gradient diagnostic max_batches must be positive")
    shared_named = _named_trainable_parameters(model, "main_branch")
    weak_named = _named_trainable_parameters(model, "right_branch")
    if not shared_named or not weak_named:
        raise ValueError(
            "gradient diagnostics require trainable main_branch and "
            "right_branch parameters"
        )
    shared_parameters = [parameter for _name, parameter in shared_named]
    weak_parameters = [parameter for _name, parameter in weak_named]
    shared_count = sum(parameter.numel() for parameter in shared_parameters)
    weak_count = sum(parameter.numel() for parameter in weak_parameters)
    model_modes = _module_modes(model)
    criterion_modes = _module_modes(criterion)
    model.eval()
    if isinstance(criterion, torch.nn.Module):
        criterion.eval()
    totals = {
        "shared_full_norm_per_sqrt_parameter": 0.0,
        "shared_dominant_norm_per_sqrt_parameter": 0.0,
        "shared_weak_norm_per_sqrt_parameter": 0.0,
        "weak_branch_norm_per_sqrt_parameter": 0.0,
        "shared_cosine_weak_dominant": 0.0,
        "shared_cosine_weak_full": 0.0,
    }
    sample_count = 0
    batch_count = 0
    try:
        with torch.enable_grad():
            for batch_index, ((x_left, x_right), targets) in enumerate(loader):
                if batch_index >= int(max_batches):
                    break
                x_left = x_left.to(device)
                x_right = x_right.to(device)
                targets = targets.to(device)
                batch_size = int(targets.size(0))

                mode_gradients = {}
                weak_branch_gradients = None
                for name, mode in (
                    ("full", FULL_MODE),
                    ("dominant", DOMINANT_ONLY_MODE),
                    ("weak", WEAK_ONLY_MODE),
                ):
                    logits = model(x_left, x_right, **mode.kwargs())
                    loss = _loss(criterion, logits, targets)
                    parameters = (
                        shared_parameters + weak_parameters
                        if name == "weak"
                        else shared_parameters
                    )
                    gradients = _gradients(loss, parameters)
                    mode_gradients[name] = gradients[: len(shared_parameters)]
                    if name == "weak":
                        weak_branch_gradients = gradients[
                            len(shared_parameters) :
                        ]

                values = {
                    "shared_full_norm_per_sqrt_parameter": (
                        _norm(mode_gradients["full"]) / math.sqrt(shared_count)
                    ),
                    "shared_dominant_norm_per_sqrt_parameter": (
                        _norm(mode_gradients["dominant"])
                        / math.sqrt(shared_count)
                    ),
                    "shared_weak_norm_per_sqrt_parameter": (
                        _norm(mode_gradients["weak"]) / math.sqrt(shared_count)
                    ),
                    "weak_branch_norm_per_sqrt_parameter": (
                        _norm(weak_branch_gradients) / math.sqrt(weak_count)
                    ),
                    "shared_cosine_weak_dominant": _cosine(
                        mode_gradients["weak"], mode_gradients["dominant"]
                    ),
                    "shared_cosine_weak_full": _cosine(
                        mode_gradients["weak"], mode_gradients["full"]
                    ),
                }
                for key, value in values.items():
                    totals[key] += value * batch_size
                sample_count += batch_size
                batch_count += 1
    finally:
        _restore_module_modes(model_modes)
        _restore_module_modes(criterion_modes)
    if not sample_count:
        raise ValueError("gradient diagnostic loader must not be empty")
    return {
        key: value / sample_count for key, value in totals.items()
    } | {
        "sample_count": sample_count,
        "batch_count": batch_count,
        "shared_parameter_count": shared_count,
        "weak_parameter_count": weak_count,
    }

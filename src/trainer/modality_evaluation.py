"""Non-invasive multimodal validation used by phase controllers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.trainer.validation_control import (
    ModalityEvaluationResult,
    ModeMetrics,
    PerExampleModeLosses,
    PerExampleModeCorrectness,
)


@dataclass(frozen=True)
class ModalityMode:
    enable_left_branch: bool
    enable_right_branch: bool
    left_branch_intervention: str | None = None
    right_branch_intervention: str | None = None

    def kwargs(self):
        return {
            "enable_left_branch": self.enable_left_branch,
            "enable_right_branch": self.enable_right_branch,
            "left_branch_intervention": self.left_branch_intervention,
            "right_branch_intervention": self.right_branch_intervention,
        }


FULL_MODE = ModalityMode(True, True)
DOMINANT_ONLY_MODE = ModalityMode(
    True, False, right_branch_intervention="deactivation"
)
WEAK_ONLY_MODE = ModalityMode(
    False, True, left_branch_intervention="deactivation"
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


def _base_classification_loss(criterion):
    current = criterion
    for _ in range(4):
        if isinstance(
            current, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)
        ):
            return current
        current = getattr(current, "criterion", None)
        if current is None:
            break
    raise TypeError(
        "paired validation uncertainty requires CrossEntropyLoss or NLLLoss"
    )


def _per_example_loss(criterion, logits, targets):
    base = _base_classification_loss(criterion)
    if isinstance(base, torch.nn.CrossEntropyLoss):
        contributions = F.cross_entropy(
            logits,
            targets,
            weight=base.weight,
            ignore_index=base.ignore_index,
            reduction="none",
            label_smoothing=base.label_smoothing,
        )
    else:
        contributions = F.nll_loss(
            logits,
            targets,
            weight=base.weight,
            ignore_index=base.ignore_index,
            reduction="none",
        )
    if base.weight is None:
        weights = torch.ones_like(contributions)
    else:
        weights = base.weight[targets]
    return contributions, weights


def _calibration_summary(
    nll_values,
    brier_values,
    confidence_values,
    correctness_values,
    *,
    num_bins=15,
):
    nll = torch.cat(nll_values).to(torch.float64)
    brier = torch.cat(brier_values).to(torch.float64)
    confidence = torch.cat(confidence_values).to(torch.float64)
    correct = torch.cat(correctness_values).to(torch.bool)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(num_bins):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        in_bin = confidence.ge(lower) & confidence.le(upper)
        if index:
            in_bin &= confidence.gt(lower)
        count = int(in_bin.sum())
        if not count:
            continue
        bin_confidence = confidence[in_bin].mean()
        bin_accuracy = correct[in_bin].to(torch.float64).mean()
        ece += (count / confidence.numel()) * torch.abs(
            bin_confidence - bin_accuracy
        )
    incorrect = ~correct
    mean_incorrect_confidence = (
        float(confidence[incorrect].mean())
        if bool(incorrect.any())
        else None
    )
    return {
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": float(ece),
        "mean_confidence": float(confidence.mean()),
        "mean_incorrect_confidence": mean_incorrect_confidence,
    }


def _evaluate_mode(model, criterion, loader, device, mode, max_batches=None):
    correct = 0
    sample_count = 0
    loss_contributions = []
    sample_weights = []
    correctness = []
    nll_values = []
    brier_values = []
    confidence_values = []
    for batch_index, ((x_left, x_right), targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x_left = x_left.to(device)
        x_right = x_right.to(device)
        targets = targets.to(device)
        logits = model(x_left, x_right, **mode.kwargs())
        contributions, weights = _per_example_loss(
            criterion, logits, targets
        )
        batch_size = targets.size(0)
        batch_correctness = logits.argmax(dim=1) == targets
        probabilities = torch.softmax(logits, dim=1)
        nll_values.append(
            F.cross_entropy(logits, targets, reduction="none").detach().cpu()
        )
        one_hot = F.one_hot(
            targets, num_classes=logits.shape[1]
        ).to(probabilities.dtype)
        brier_values.append(
            ((probabilities - one_hot) ** 2)
            .sum(dim=1)
            .detach()
            .cpu()
        )
        confidence_values.append(
            probabilities.max(dim=1).values.detach().cpu()
        )
        correct += int(batch_correctness.sum())
        correctness.append(batch_correctness.detach().cpu())
        sample_count += batch_size
        loss_contributions.append(contributions.detach().cpu())
        sample_weights.append(weights.detach().cpu())
    if sample_count == 0:
        raise ValueError("validation loader must not be empty")
    contributions = torch.cat(loss_contributions).to(torch.float64)
    weights = torch.cat(sample_weights).to(torch.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("validation criterion has zero total sample weight")
    normalized_losses = contributions * (sample_count / weight_sum)
    calibration = _calibration_summary(
        nll_values,
        brier_values,
        confidence_values,
        correctness,
    )
    return (
        ModeMetrics(
            loss=float(contributions.sum() / weights.sum()),
            accuracy=correct / sample_count,
            **calibration,
        ),
        tuple(float(value) for value in normalized_losses.tolist()),
        tuple(int(value) for value in torch.cat(correctness).tolist()),
    )


def evaluate_single_mode(
    model, criterion, loader, device, *, mode, max_batches=None
):
    """Evaluate one mode without changing parameters or module modes."""
    if max_batches is not None and int(max_batches) < 1:
        raise ValueError("max_batches must be positive when configured")
    model_modes = _module_modes(model)
    criterion_modes = _module_modes(criterion)
    model.eval()
    if isinstance(criterion, torch.nn.Module):
        criterion.eval()
    try:
        with torch.no_grad():
            metrics, _losses, _correctness = _evaluate_mode(
                model, criterion, loader, device, mode, max_batches
            )
    finally:
        _restore_module_modes(model_modes)
        _restore_module_modes(criterion_modes)
    return metrics


def evaluate_modalities(
    model,
    criterion,
    validation_loader,
    device,
    *,
    intervention_mode,
    phase_epoch,
    global_epoch,
    global_step,
    max_batches=None,
):
    """Evaluate all branch modes while preserving parameters and module modes."""
    if max_batches is not None and int(max_batches) < 1:
        raise ValueError("max_batches must be positive when configured")
    model_modes = _module_modes(model)
    criterion_modes = _module_modes(criterion)
    model.eval()
    if isinstance(criterion, torch.nn.Module):
        criterion.eval()
    try:
        with torch.no_grad():
            evaluated = {}
            for mode in (
                FULL_MODE,
                DOMINANT_ONLY_MODE,
                WEAK_ONLY_MODE,
                intervention_mode,
            ):
                if mode not in evaluated:
                    evaluated[mode] = _evaluate_mode(
                        model, criterion, validation_loader, device, mode,
                        max_batches,
                    )
            full, full_losses, full_correct = evaluated[FULL_MODE]
            dominant_only, dominant_losses, dominant_correct = evaluated[
                DOMINANT_ONLY_MODE
            ]
            weak_only, weak_losses, weak_correct = evaluated[WEAK_ONLY_MODE]
            intervention, intervention_losses, intervention_correct = evaluated[
                intervention_mode
            ]
    finally:
        _restore_module_modes(model_modes)
        _restore_module_modes(criterion_modes)
    return ModalityEvaluationResult(
        full=full,
        dominant_only=dominant_only,
        weak_only=weak_only,
        intervention=intervention,
        phase_epoch=int(phase_epoch),
        global_epoch=int(global_epoch),
        global_step=int(global_step),
        per_example_losses=PerExampleModeLosses(
            full=full_losses,
            dominant_only=dominant_losses,
            weak_only=weak_losses,
            intervention=intervention_losses,
        ),
        per_example_correctness=PerExampleModeCorrectness(
            full=full_correct,
            dominant_only=dominant_correct,
            weak_only=weak_correct,
            intervention=intervention_correct,
        ),
    )

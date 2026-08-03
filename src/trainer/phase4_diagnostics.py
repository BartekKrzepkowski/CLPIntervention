"""Phase-4 compatibility diagnostics against the pre-Phase-4 anchor."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.trainer.modality_evaluation import (
    WEAK_ONLY_MODE,
    evaluate_single_mode,
)
from src.trainer.validation_control import ModeMetrics


_LEFT_PREFIX = "left_branch."
_RIGHT_PREFIX = "right_branch."


@dataclass(frozen=True)
class Phase4TrainabilityState:
    """Trainability applied at one local Phase-4 epoch."""

    stage: str
    phase_epoch: int
    shared_only_epochs: int
    trainable_left: bool
    trainable_right: bool
    trainable_shared: bool

    @property
    def shared_only_active(self):
        return self.stage == "shared_only"


def configure_phase4_trainability(model, phase_epoch, shared_only_epochs):
    """Optionally adapt shared downstream before unfreezing the encoders.

    The optimizer must be constructed before calling this function so that it
    already contains all Phase-4 parameters. During the shared-only prefix the
    encoders still participate in the forward pass, but their parameters and
    BatchNorm buffers remain frozen.
    """
    phase_epoch = int(phase_epoch)
    shared_only_epochs = int(shared_only_epochs)
    if phase_epoch < 1:
        raise ValueError("phase_epoch must be positive during Phase-4 training")
    if shared_only_epochs < 0:
        raise ValueError("shared_only_epochs must be non-negative")

    shared_only = phase_epoch <= shared_only_epochs
    model.requires_grad_(True)
    model.train()
    if shared_only:
        model.left_branch.requires_grad_(False)
        model.right_branch.requires_grad_(False)
        model.left_branch.eval()
        model.right_branch.eval()

    return Phase4TrainabilityState(
        stage="shared_only" if shared_only else "full",
        phase_epoch=phase_epoch,
        shared_only_epochs=shared_only_epochs,
        trainable_left=not shared_only,
        trainable_right=not shared_only,
        trainable_shared=True,
    )


def _clone_cpu_state(state):
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


@dataclass(frozen=True)
class Phase4HybridMetrics:
    """Weak-only performance of the two anchor/current hybrid models."""

    current_right_anchor_shared: ModeMetrics
    anchor_right_current_shared: ModeMetrics

    def state_dict(self):
        return {
            "current_right_anchor_shared": (
                self.current_right_anchor_shared.state_dict()
            ),
            "anchor_right_current_shared": (
                self.anchor_right_current_shared.state_dict()
            ),
        }


@dataclass(frozen=True)
class Phase4HybridAnchor:
    """Pre-Phase-4 right encoder and shared downstream state.

    The shared downstream contains every state entry outside ``left_branch``
    and ``right_branch``. For S-ResNet-18 this includes ``main_branch`` and
    the final classifier, which are both frozen by relative-unimodal-parity
    Phase 3.
    """

    right_state: dict[str, torch.Tensor]
    shared_state: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model):
        state = model.state_dict()
        right = {
            name: value
            for name, value in state.items()
            if name.startswith(_RIGHT_PREFIX)
        }
        shared = {
            name: value
            for name, value in state.items()
            if not name.startswith((_LEFT_PREFIX, _RIGHT_PREFIX))
        }
        if not right:
            raise ValueError("Phase-4 diagnostic requires right_branch state")
        if not shared:
            raise ValueError("Phase-4 diagnostic requires shared downstream state")
        return cls(_clone_cpu_state(right), _clone_cpu_state(shared))

    def state_dict(self):
        return {
            "version": 1,
            "right_state": _clone_cpu_state(self.right_state),
            "shared_state": _clone_cpu_state(self.shared_state),
        }

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get("version", 0)) != 1:
            raise ValueError("Unsupported Phase4HybridAnchor state version")
        return cls(
            _clone_cpu_state(state["right_state"]),
            _clone_cpu_state(state["shared_state"]),
        )


@contextmanager
def _temporary_state(model, replacement):
    live_state = model.state_dict()
    missing = sorted(set(replacement) - set(live_state))
    if missing:
        raise ValueError(f"Diagnostic anchor has unknown state keys: {missing}")
    originals = {}
    with torch.no_grad():
        for name, value in replacement.items():
            target = live_state[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"Diagnostic anchor shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            originals[name] = target.detach().clone()
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            restored_state = model.state_dict()
            for name, value in originals.items():
                restored_state[name].copy_(value)


def evaluate_phase4_hybrids(
    model,
    anchor,
    criterion,
    validation_loader,
    device,
    *,
    max_batches=None,
):
    """Separate right-encoder drift from shared-downstream drift.

    Both measurements use weak-only validation proper. Temporary state swaps
    include parameters and buffers and are fully undone before returning.
    """
    with _temporary_state(model, anchor.shared_state):
        current_right_anchor_shared = evaluate_single_mode(
            model,
            criterion,
            validation_loader,
            device,
            mode=WEAK_ONLY_MODE,
            max_batches=max_batches,
        )
    with _temporary_state(model, anchor.right_state):
        anchor_right_current_shared = evaluate_single_mode(
            model,
            criterion,
            validation_loader,
            device,
            mode=WEAK_ONLY_MODE,
            max_batches=max_batches,
        )
    return Phase4HybridMetrics(
        current_right_anchor_shared=current_right_anchor_shared,
        anchor_right_current_shared=anchor_right_current_shared,
    )

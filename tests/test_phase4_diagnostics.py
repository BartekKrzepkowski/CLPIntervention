from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from scripts.python_new.run_single import _add_phase4_lr_warmup
from src.trainer.phase4_diagnostics import (
    Phase4HybridAnchor,
    configure_phase4_trainability,
    evaluate_phase4_hybrids,
)


class _PairDataset(Dataset):
    def __init__(self):
        self.right = torch.tensor([[1.0], [-1.0]])
        self.targets = torch.tensor([0, 1])

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (torch.zeros(1), self.right[index]), self.targets[index]


class _HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left_branch = torch.nn.Linear(1, 2, bias=False)
        self.right_branch = torch.nn.Linear(1, 2, bias=False)
        self.main_branch = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer("shared_scale", torch.tensor(1.0))
        with torch.no_grad():
            self.left_branch.weight.zero_()
            self.right_branch.weight.copy_(torch.tensor([[1.0], [-1.0]]))
            self.main_branch.weight.copy_(torch.eye(2))

    def forward(
        self,
        left,
        right,
        *,
        enable_left_branch=True,
        enable_right_branch=True,
        left_branch_intervention=None,
        right_branch_intervention=None,
    ):
        if enable_left_branch:
            left_features = self.left_branch(left)
        elif left_branch_intervention == "deactivation":
            left_features = left.new_zeros((left.shape[0], 2))
        else:
            raise ValueError("unsupported left intervention")
        if enable_right_branch:
            right_features = self.right_branch(right)
        elif right_branch_intervention == "deactivation":
            right_features = right.new_zeros((right.shape[0], 2))
        else:
            raise ValueError("unsupported right intervention")
        return self.main_branch(
            (left_features + right_features) * self.shared_scale
        )


def test_phase4_hybrid_diagnostic_separates_and_restores_state_and_modes():
    model = _HybridModel()
    anchor = Phase4HybridAnchor.capture(model)
    restored_anchor = Phase4HybridAnchor.from_state_dict(anchor.state_dict())
    with torch.no_grad():
        model.right_branch.weight.copy_(torch.tensor([[-1.0], [1.0]]))
        model.main_branch.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        model.shared_scale.fill_(2.0)
    model.train()
    model.right_branch.eval()
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    modes_before = {
        name: module.training for name, module in model.named_modules()
    }

    result = evaluate_phase4_hybrids(
        model,
        restored_anchor,
        torch.nn.CrossEntropyLoss(),
        DataLoader(_PairDataset(), batch_size=2, shuffle=False),
        torch.device("cpu"),
    )

    assert result.current_right_anchor_shared.accuracy == 0.0
    assert result.anchor_right_current_shared.accuracy == 0.0
    assert all(
        torch.equal(value, before[name])
        for name, value in model.state_dict().items()
    )
    assert {
        name: module.training for name, module in model.named_modules()
    } == modes_before
    assert "shared_scale" in restored_anchor.shared_state


def test_phase4_warmup_is_per_step_and_uses_phase4_metric_prefix():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.6)
    epoch_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.5
    )
    scheduler = _add_phase4_lr_warmup(
        optimizer,
        epoch_scheduler,
        epochs=4,
        start_factor=0.1,
        steps_per_epoch=2,
    )

    assert optimizer.param_groups[0]["lr"] == 0.06
    assert scheduler.metric_prefix == "phase4"
    for _ in range(8):
        optimizer.step()
        scheduler.step_batch()
    assert scheduler.completed_steps == 8
    assert optimizer.param_groups[0]["lr"] == 0.6


def test_phase4_shared_only_prefix_freezes_encoders_then_unfreezes_all():
    model = _HybridModel()

    shared_only = configure_phase4_trainability(
        model, phase_epoch=4, shared_only_epochs=4
    )
    assert shared_only.stage == "shared_only"
    assert not any(p.requires_grad for p in model.left_branch.parameters())
    assert not any(p.requires_grad for p in model.right_branch.parameters())
    assert all(p.requires_grad for p in model.main_branch.parameters())
    assert not model.left_branch.training
    assert not model.right_branch.training
    assert model.main_branch.training

    full = configure_phase4_trainability(
        model, phase_epoch=5, shared_only_epochs=4
    )
    assert full.stage == "full"
    assert all(p.requires_grad for p in model.parameters())
    assert all(module.training for module in model.modules())

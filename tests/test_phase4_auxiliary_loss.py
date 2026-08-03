from __future__ import annotations

from collections import defaultdict

from omegaconf import OmegaConf
import pytest
import torch

from src.modules.losses import ClassificationLoss
from src.trainer.trainer_classification_mm_clp import (
    TrainerClassification,
    phase4_auxiliary_loss_weights,
)


def test_phase4_auxiliary_loss_is_opt_in():
    assert phase4_auxiliary_loss_weights(OmegaConf.create({})) == (0.0, 0.0)


def test_phase4_auxiliary_loss_reads_asymmetric_weights():
    config = OmegaConf.create(
        {
            "phase4_auxiliary_loss": {
                "enabled": True,
                "weak_weight": 1.0,
                "dominant_weight": 0.0,
            }
        }
    )
    assert phase4_auxiliary_loss_weights(config) == (1.0, 0.0)


def test_phase4_auxiliary_loss_rejects_negative_weights():
    config = OmegaConf.create(
        {
            "phase4_auxiliary_loss": {
                "enabled": True,
                "weak_weight": -0.1,
            }
        }
    )
    with pytest.raises(ValueError, match="non-negative"):
        phase4_auxiliary_loss_weights(config)


class _RecordingBimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.right_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.shared_calls = []

    def classify_encoded_modalities(self, features_left, features_right):
        self.shared_calls.append((features_left, features_right))
        score = features_left[:, 0] + features_right[:, 0]
        return torch.stack((score, -score), dim=1)

    def forward(
        self,
        x_left,
        x_right,
        *,
        left_branch_intervention=None,
        right_branch_intervention=None,
        enable_left_branch=True,
        enable_right_branch=True,
        return_features=False,
    ):
        self.calls.append(
            {
                "enable_left_branch": enable_left_branch,
                "enable_right_branch": enable_right_branch,
                "left_branch_intervention": left_branch_intervention,
                "right_branch_intervention": right_branch_intervention,
            }
        )
        left_features = x_left * 0.0
        right_features = x_right * 0.0
        if enable_left_branch:
            left_features = self.left_scale * x_left
        if enable_right_branch:
            right_features = self.right_scale * x_right
        logits = self.classify_encoded_modalities(
            left_features, right_features
        )
        if return_features:
            return logits, left_features, right_features
        return logits


@pytest.mark.gpu
def test_zero_dominant_weight_skips_dominant_only_training_forward():
    model = _RecordingBimodalModel()
    trainer = TrainerClassification(
        model=model,
        criterion=ClassificationLoss("ce"),
        loaders={},
        optim=None,
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    config = OmegaConf.create(
        {
            "phase": 4,
            "extra": {
                "left_branch_intervention": None,
                "right_branch_intervention": None,
                "enable_left_branch": True,
                "enable_right_branch": True,
            },
            "phase4_auxiliary_loss": {
                "enabled": True,
                "weak_weight": 1.0,
                "dominant_weight": 0.0,
            },
        }
    )
    model.train()
    loss, evaluators = trainer.compute_loss(
        torch.tensor([[1.0], [0.5]]),
        torch.tensor([[0.25], [1.0]]),
        torch.tensor([0, 1]),
        config,
    )
    loss.backward()

    assert len(model.calls) == 1
    assert model.calls[0]["enable_left_branch"] is True
    assert len(model.shared_calls) == 2
    assert torch.count_nonzero(model.shared_calls[1][0]).item() == 0
    assert torch.equal(model.shared_calls[1][1], model.right_scale * torch.tensor([[0.25], [1.0]]))
    assert model.right_scale.grad is not None
    assert evaluators["phase4_loss/dominant_weight"] == 0.0
    assert "phase4_loss/dominant_only" not in evaluators

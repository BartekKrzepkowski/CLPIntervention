from collections import defaultdict
import random

import numpy as np

import torch

from src.modules.aux_modules import DeadReLU
from src.modules.losses import BalancePenaltyLoss, MSESoftmaxLoss
from src.utils.utils_optim import configure_optimizer
from src.utils.utils_trainer import (
    adjust_evaluators_pre_log,
    load_branch,
    load_training_checkpoint,
    manual_seed,
    restore_rng_state,
    save_training_checkpoint,
)



def test_manual_seed_controls_python_numpy_and_torch_rngs():
    manual_seed(19, torch.device("cpu"))
    first = (random.random(), np.random.random(), torch.rand(3))
    manual_seed(19, torch.device("cpu"))
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


def test_metrics_remain_unrounded_for_loggers():
    raw = {"loss": 1.0}
    averaged = adjust_evaluators_pre_log(raw, denom=3.0)

    assert averaged["loss"] == 1.0 / 3.0
    assert averaged["loss"] != round(1.0 / 3.0, 4)
    assert adjust_evaluators_pre_log(raw, denom=3.0, round_at=4)["loss"] == 0.3333


def test_optimizer_configuration_does_not_mutate_input_mapping():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    options = {"lr": 0.1, "weight_decay": 0.01}
    optimizer = configure_optimizer(torch.optim.SGD, model, options)

    assert options == {"lr": 0.1, "weight_decay": 0.01}
    assert len(optimizer.param_groups) == 2
    assert [group["weight_decay_enabled"] for group in optimizer.param_groups] == [
        True,
        False,
    ]


def test_mse_softmax_uses_runtime_class_count_and_trainer_contract():
    criterion = MSESoftmaxLoss()
    loss, metrics = criterion(torch.randn(4, 3, requires_grad=True), torch.tensor([0, 1, 2, 1]))
    assert loss.ndim == 0
    assert set(metrics) == {"loss", "acc"}


def test_balance_penalty_skips_second_order_regularizer_during_evaluation():
    model = torch.nn.Module()
    model.left_branch = torch.nn.Linear(2, 2)
    model.right_branch = torch.nn.Linear(2, 2)
    criterion = BalancePenaltyLoss(model, "ce", num_classes=2, weight=torch.tensor([1.0, 2.0]))

    class FailingRegularizer(torch.nn.Module):
        def forward(self, predictions):
            raise AssertionError("regularizer should not run during evaluation")

    criterion.regularizer = FailingRegularizer()
    criterion.eval()
    with torch.no_grad():
        loss, metrics = criterion(torch.randn(2, 2), torch.tensor([0, 1]))
    assert loss.ndim == 0
    assert set(metrics) == {"loss", "acc"}
    torch.testing.assert_close(criterion.criterion.criterion.weight, torch.tensor([1.0, 2.0]))


def test_load_branch_accepts_full_bimodal_checkpoint(tmp_path):
    left = torch.nn.Linear(3, 2)
    right = torch.nn.Linear(3, 2)
    state_dict = {
        **{f"left_branch.{name}": value.clone() for name, value in left.state_dict().items()},
        **{f"right_branch.{name}": value.clone() for name, value in right.state_dict().items()},
    }
    checkpoint = tmp_path / "model.pth"
    torch.save(state_dict, checkpoint)

    restored = torch.nn.Linear(3, 2)
    load_branch(restored, checkpoint, "left_branch", device=torch.device("cpu"))
    for expected, actual in zip(left.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)



def test_training_checkpoint_restores_optimizer_scheduler_epoch_and_rng(tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.8
    )
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint = tmp_path / "training.pth"
    save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint,
        next_epoch=4,
        global_step=16,
        metadata={
            "kind": "proper",
            "protocol_manifest": {
                "version": 1,
                "loader": {"batch_size": 2},
            },
        },
        diagnostics_state={"run_stats": {"version": 1, "marker": 7}},
        phase_state={"detector": {"bad_checks": 2}},
        loader_state={"train_generator_state": torch.tensor([1, 2, 3])},
        phase_epoch=3,
    )
    expected_random = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(), lr=0.9, momentum=0.9
    )
    restored_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        restored_optimizer, lr_lambda=lambda _: 0.5
    )
    state = load_training_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        device=torch.device("cpu"),
    )
    restore_rng_state(state["rng_state"])

    assert state["is_training_checkpoint"] is True
    assert state["next_epoch"] == 4
    assert state["global_step"] == 16
    assert state["metadata"] == {
        "kind": "proper",
        "protocol_manifest": {
            "version": 1,
            "loader": {"batch_size": 2},
        },
    }
    assert state["diagnostics_state"]["run_stats"]["marker"] == 7
    assert state["phase_state"]["detector"]["bad_checks"] == 2
    assert state["phase_epoch"] == 3
    torch.testing.assert_close(
        state["loader_state"]["train_generator_state"],
        torch.tensor([1, 2, 3]),
    )
    assert not list(tmp_path.glob("*.tmp-*"))
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored_scheduler.lr_lambdas[0](0) == 0.8
    expected_momentum = next(iter(optimizer.state.values()))["momentum_buffer"]
    restored_momentum = next(iter(restored_optimizer.state.values()))[
        "momentum_buffer"
    ]
    torch.testing.assert_close(restored_momentum, expected_momentum)
    assert random.random() == expected_random
    assert np.random.random() == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)

def test_dead_relu_metric_tolerates_a_disabled_branch_without_activations():
    metric = DeadReLU(torch.nn.Sequential(torch.nn.ReLU()), is_left_branch=True, is_able=False)
    metric.at_the_epoch_end("train", max_dataset=4, step=0)

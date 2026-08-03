from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.trainer.trainer_classification_mm_clp import (
    TrainerClassification,
    _measurement_batch_indices,
)


class PairDataset(Dataset):
    transform2 = None

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (torch.ones(4) * index, torch.ones(4)), index % 2


class TinyBimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, x1, x2, **kwargs):
        return self.classifier(torch.cat((x1, x2), dim=1))


class RecordingCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, predictions, targets):
        self.calls.append((torch.is_grad_enabled(), self.training))
        loss = torch.nn.functional.cross_entropy(predictions, targets)
        return loss, {"loss": loss.item(), "acc": 0.0}


class Logger:
    def __init__(self):
        self.calls = []

    def log_scalars(self, values, step):
        self.calls.append((dict(values), step))


def test_validation_runs_without_autograd_and_with_eval_criterion(tmp_path):
    dataset = PairDataset()
    loaders = {
        "train": DataLoader(dataset, batch_size=2),
        "test_proper": DataLoader(dataset, batch_size=2),
        "test_blurred": DataLoader(dataset, batch_size=2),
    }
    model = TinyBimodalModel()
    criterion = RecordingCriterion()
    trainer = TrainerClassification(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    trainer.logger = Logger()
    trainer.save_path = lambda step: str(tmp_path / f"{step}.pth")
    config = SimpleNamespace(
        whether_disable_tqdm=True,
        extra={
            "left_branch_intervention": None,
            "right_branch_intervention": None,
            "enable_left_branch": True,
            "enable_right_branch": True,
        },
        clip_value=0.0,
        run_stats_multi=0,
        fim_measurements_per_epoch=0,
        stiffness_multi=0,
        rank_multi=0,
        log_multi=1,
        logger_config={"hyperparameters": {"type_names": {"scheduler": "multiplicative"}}},
        protocol_manifest=OmegaConf.create(
            {"version": 1, "loader": {"batch_size": 2}}
        ),
    )

    trainer.run_loop(0, 1, config)

    assert criterion.calls == [(True, True), (False, False), (False, False)]
    assert model.training is False
    checkpoint = torch.load(
        tmp_path / "epoch_0.pth", map_location="cpu", weights_only=True
    )
    assert checkpoint["format"] == "clpintervention.training"
    assert checkpoint["version"] == 5
    assert checkpoint["metadata"]["protocol_manifest"] == {
        "version": 1,
        "loader": {"batch_size": 2},
    }
    assert set(checkpoint) >= {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "next_epoch",
        "global_step",
        "rng_state",
    }


def test_phase4_can_stop_at_the_reference_training_accuracy(tmp_path):
    model = TinyBimodalModel()
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={},
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    trainer.save_path = lambda step: str(tmp_path / f"{step}.pth")
    train_accuracies = iter((0.4, 0.91, 0.95))
    train_epochs = []

    def fake_run_epoch(phase, config):
        if phase == "train":
            train_epochs.append(trainer.epoch)
            return {"epoch_acc/train": next(train_accuracies)}
        return {}

    trainer.run_epoch = fake_run_epoch
    trainer.run_loop(
        0, 5, SimpleNamespace(whether_disable_tqdm=True), target_train_acc=0.9
    )

    assert train_epochs == [0, 1]
    assert trainer.epoch == 1


def test_phase4_optimizer_overrides_are_applied_only_when_requested():
    model = TinyBimodalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.6, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={},
        optim=optimizer,
        lr_scheduler=scheduler,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )

    trainer._apply_phase4_optimizer_overrides(
        SimpleNamespace(phase4_weight_decay=5e-4, phase4_lr_lambda=0.98)
    )

    assert optimizer.param_groups[0]["weight_decay"] == 5e-4
    assert scheduler.lr_lambdas[0](0) == 0.98


def test_phase3_linear_lr_warmup_reaches_base_lr_and_roundtrips_state():
    from scripts.python_new.run_single import _add_phase3_lr_warmup

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.6)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 1.0
    )
    scheduler = _add_phase3_lr_warmup(
        optimizer,
        scheduler,
        epochs=4,
        start_factor=0.1,
        steps_per_epoch=2,
    )
    observed = [optimizer.param_groups[0]["lr"]]
    for step in range(3):
        optimizer.step()
        scheduler.step_batch()
        observed.append(optimizer.param_groups[0]["lr"])

    assert observed[0] == pytest.approx(0.06)
    assert scheduler.get_last_lr() == pytest.approx(
        [optimizer.param_groups[0]["lr"]]
    )
    assert observed == sorted(observed)
    assert observed[-1] < 0.6

    restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
    restored_optimizer = torch.optim.SGD([restored_parameter], lr=0.6)
    restored_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        restored_optimizer, lr_lambda=lambda _: 1.0
    )
    restored_scheduler = _add_phase3_lr_warmup(
        restored_optimizer,
        restored_scheduler,
        epochs=4,
        start_factor=0.1,
        steps_per_epoch=2,
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()

    for step in range(3, 8):
        optimizer.step()
        scheduler.step_batch()
        restored_optimizer.step()
        restored_scheduler.step_batch()

    assert scheduler.completed_steps == 8
    assert restored_scheduler.completed_steps == 8
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.6)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.6)
    assert scheduler.state_dict() == restored_scheduler.state_dict()


def test_stage_specs_apply_interventions_and_right_transform(monkeypatch):
    import src.trainer.trainer_classification_mm_clp as trainer_module

    dataset = PairDataset()
    loaders = {
        "train": DataLoader(dataset, batch_size=2),
        "test_proper": DataLoader(dataset, batch_size=2),
        "test_blurred": DataLoader(dataset, batch_size=2),
    }
    model = TinyBimodalModel()
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders=loaders,
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    monkeypatch.setitem(
        trainer_module.TRANSFORMS_BLURRED_RIGHT_NAME_MAP,
        "fake_dataset",
        lambda overlap, resize_factor, normalization_profile=None: (
            "blurred", overlap, resize_factor
        ),
    )
    monkeypatch.setitem(
        trainer_module.TRANSFORMS_PROPER_RIGHT_NAME_MAP,
        "fake_dataset",
        lambda overlap, normalization_profile=None: ("proper", overlap),
    )
    config = SimpleNamespace(
        overlap=0.125,
        resize_factor=0.5,
        logger_config={"hyperparameters": {"type_names": {"dataset": "fake_dataset"}}},
    )

    trainer._apply_stage(config, trainer_module.PHASE_STAGES[1])
    assert config.kind == "blurred"
    assert config.extra["enable_left_branch"] is True
    assert dataset.transform2 == ("blurred", 0.125, 0.5)

    trainer._apply_stage(config, trainer_module.PHASE_STAGES[3])
    assert config.kind == "proper"
    assert config.extra["enable_left_branch"] is False
    assert config.extra["left_branch_intervention"] == "deactivation"
    assert dataset.transform2 == ("proper", 0.125)

    trainer._apply_stage(config, trainer_module.PHASE_STAGES[4])
    assert config.extra["enable_left_branch"] is True
    assert config.extra["enable_right_branch"] is True
    assert dataset.transform2 == ("proper", 0.125)

    trainer._apply_stage(config, trainer_module.PRETRAIN_STAGES["right_proper"])
    assert config.kind == "proper"
    assert config.extra["enable_left_branch"] is False
    assert config.extra["enable_right_branch"] is True


def test_all_at_once_uses_the_four_shared_stage_specs():
    import src.trainer.trainer_classification_mm_clp as trainer_module

    trainer = object.__new__(TrainerClassification)
    calls = []
    trainer.manual_seed = lambda config: None
    trainer.at_exp_start = lambda config: None
    trainer._run_stage = (
        lambda config, stage, start, end, close_logger, target_train_acc=None:
        calls.append((stage, start, end, close_logger, target_train_acc))
    )

    class ClosingLogger:
        closed = False

        def close(self):
            self.closed = True

    trainer.logger = ClosingLogger()
    config = SimpleNamespace(
        phase1_starts_at_epoch=0,
        phase1_ends_at_epoch=10,
        phase2_ends_at_epoch=20,
        phase3_ends_at_epoch=25,
        phase4_ends_at_epoch=40,
        phase4_target_train_acc=0.9,
    )

    trainer.run_all_at_once(config)

    assert calls == [
        (trainer_module.PHASE_STAGES[1], 0, 10, False, None),
        (trainer_module.PHASE_STAGES[2], 10, 20, False, None),
        (trainer_module.PHASE_STAGES[3], 20, 25, False, None),
        (trainer_module.PHASE_STAGES[4], 25, 40, False, 0.9),
    ]
    assert trainer.logger.closed is True



def test_all_at_once_resume_skips_completed_phases_and_continues_mid_phase():
    import src.trainer.trainer_classification_mm_clp as trainer_module

    trainer = object.__new__(TrainerClassification)
    trainer._initialize_run = lambda config: None
    trainer._apply_phase4_optimizer_overrides = lambda config: None
    calls = []
    trainer._run_stage = (
        lambda config, stage, start, end, close_logger, target_train_acc=None:
        calls.append((stage, start, end, target_train_acc))
    )
    trainer.logger = type("Logger", (), {"close": lambda self: None})()
    config = SimpleNamespace(
        exp_starts_at_epoch=22,
        phase1_starts_at_epoch=0,
        phase1_ends_at_epoch=10,
        phase2_ends_at_epoch=20,
        phase3_ends_at_epoch=25,
        phase4_ends_at_epoch=40,
        phase4_target_train_acc=0.9,
    )

    trainer.run_all_at_once(config)

    assert calls == [
        (trainer_module.PHASE_STAGES[3], 22, 25, None),
        (trainer_module.PHASE_STAGES[4], 25, 40, 0.9),
    ]

def test_umt_loss_distills_only_enabled_branches():
    from src.trainer.trainer_classification_mm_clp_umt import (
        TrainerClassification as UMTTrainer,
    )

    class TinyUMTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(8, 2)

        def forward(self, x_left, x_right, **kwargs):
            predictions = self.classifier(torch.cat((x_left, x_right), dim=1))
            return predictions, x_left + 1.0, x_right + 2.0

        def teacher_features(
            self,
            x_left,
            x_right,
            *,
            enable_left_branch,
            enable_right_branch,
        ):
            return (
                x_left if enable_left_branch else None,
                x_right if enable_right_branch else None,
            )

    model = TinyUMTModel()
    criterion = RecordingCriterion()
    trainer = UMTTrainer(
        model=model,
        criterion=criterion,
        loaders={},
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    x_left = torch.zeros(2, 4)
    x_right = torch.zeros(2, 4)
    targets = torch.tensor([0, 1])
    config = SimpleNamespace(
        extra={"enable_left_branch": False, "enable_right_branch": True,
               "left_branch_intervention": "deactivation",
               "right_branch_intervention": None},
        distill=0.5,
    )

    base_loss, _ = criterion(model.classifier(torch.cat((x_left, x_right), dim=1)), targets)
    loss, metrics = trainer.compute_loss(x_left, x_right, targets, config)

    expected_right_mse = torch.nn.functional.mse_loss(x_right + 2.0, x_right)
    assert torch.allclose(loss, base_loss + 0.5 * expected_right_mse)
    assert "distillation/mse_left" not in metrics
    assert metrics["distillation/mse_right"] == expected_right_mse.item()
    assert metrics["classification_loss"] == base_loss.item()
    assert metrics["distillation/loss"] == expected_right_mse.item()
    assert metrics["distillation/weighted_loss"] == (
        0.5 * expected_right_mse
    ).item()
    assert metrics["loss"] == loss.item()


def test_frozen_left_active_optimizer_excludes_only_left_encoder():
    from scripts.python_new.run_single import configure_phase_trainability

    class BranchedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left_branch = torch.nn.Linear(2, 2)
            self.right_branch = torch.nn.Linear(2, 2)
            self.main_branch = torch.nn.Linear(2, 2)

    model = BranchedModel()
    configure_phase_trainability(
        model,
        3,
        phase3_rule="local_accuracy",
        phase3_intervention="frozen_left_active",
    )

    assert all(not parameter.requires_grad for parameter in model.left_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.right_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.main_branch.parameters())
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized.isdisjoint(
        {id(parameter) for parameter in model.left_branch.parameters()}
    )

def _training_config(**overrides):
    values = {
        "whether_disable_tqdm": True,
        "extra": {
            "left_branch_intervention": None,
            "right_branch_intervention": None,
            "enable_left_branch": True,
            "enable_right_branch": True,
        },
        "clip_value": 0.0,
        "run_stats_multi": 0,
        "fim_measurements_per_epoch": 0,
        "stiffness_multi": 0,
        "rank_multi": 0,
        "log_multi": 100,
        "kind": "proper",
        "logger_config": {
            "hyperparameters": {
                "type_names": {"scheduler": "multiplicative"}
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)



def test_training_loop_records_every_optimizer_displacement():
    class RecordingRunStats:
        logger = None

        def __init__(self):
            self.recorded_steps = 0

        def record_optimizer_step(self):
            self.recorded_steps += 1

        def __call__(self, *_args):
            raise AssertionError("periodic logging should be disabled")

    dataset = PairDataset()
    model = TinyBimodalModel()
    run_stats = RecordingRunStats()
    extras = defaultdict(lambda: None)
    extras["run_stats"] = run_stats
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={"train": DataLoader(dataset, batch_size=1)},
        optim=torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.2),
        lr_scheduler=None,
        extra_modules=extras,
        device=torch.device("cpu"),
    )
    trainer.logger = Logger()
    trainer.epoch = 0

    trainer.run_epoch("train", _training_config(run_stats_multi=0))

    assert run_stats.recorded_steps == 2


def test_learning_rate_is_logged_directly_without_sample_averaging():
    dataset = PairDataset()
    model = TinyBimodalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.5
    )
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={"train": DataLoader(dataset, batch_size=1)},
        optim=optimizer,
        lr_scheduler=scheduler,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    trainer.logger = Logger()
    trainer.epoch = 0

    epoch_metrics = trainer.run_epoch("train", _training_config())

    lr_calls = [
        values for values, _step in trainer.logger.calls
        if "lr/training" in values
    ]
    assert lr_calls == [{"lr/training": 0.05, "steps/lr": 1}]
    assert not any("lr" in key for key in epoch_metrics)
    logged_steps = [step for _values, step in trainer.logger.calls]
    assert logged_steps == sorted(logged_steps)
    epoch_call = next(
        values
        for values, _step in trainer.logger.calls
        if "steps/train_epoch" in values
    )
    assert epoch_call["steps/train_epoch"] == 0


def test_phase3_warmup_steps_per_batch_without_epoch_scheduler():
    from scripts.python_new.run_single import _add_phase3_lr_warmup

    dataset = PairDataset()
    model = TinyBimodalModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    epoch_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.5
    )
    scheduler = _add_phase3_lr_warmup(
        optimizer,
        epoch_scheduler,
        epochs=1,
        start_factor=0.1,
        steps_per_epoch=2,
    )
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={"train": DataLoader(dataset, batch_size=1)},
        optim=optimizer,
        lr_scheduler=scheduler,
        extra_modules=defaultdict(lambda: None),
        device=torch.device("cpu"),
    )
    trainer.logger = Logger()
    trainer.epoch = 0

    trainer.run_epoch("train", _training_config())

    lr_used = [
        values["phase3/lr_used"]
        for values, _step in trainer.logger.calls
        if "phase3/lr_used" in values
    ]
    assert lr_used == pytest.approx([0.01, 0.055])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert epoch_scheduler.last_epoch == 0
    assert not any(
        "lr/training" in values for values, _step in trainer.logger.calls
    )


def test_fim_schedule_uses_exact_local_measurement_count_each_epoch():
    assert _measurement_batch_indices(6, 2) == {0, 3}
    assert _measurement_batch_indices(5, 2) == {0, 2}
    assert _measurement_batch_indices(3, 2) == {0, 1}
    assert _measurement_batch_indices(1, 2) == {0}
    assert _measurement_batch_indices(5, 0) == set()
    with pytest.raises(ValueError, match="non-negative"):
        _measurement_batch_indices(5, -1)

    class RecordingFIM:
        logger = None

        def __init__(self):
            self.steps = []

        def __call__(self, step, config, kind):
            self.steps.append((step, kind))

    dataset = torch.utils.data.ConcatDataset([PairDataset()] * 3)
    model = TinyBimodalModel()
    fim = RecordingFIM()
    extras = defaultdict(lambda: None)
    extras["trace_fim_train"] = fim
    trainer = TrainerClassification(
        model=model,
        criterion=RecordingCriterion(),
        loaders={"train": DataLoader(dataset, batch_size=1)},
        optim=torch.optim.SGD(model.parameters(), lr=0.1),
        lr_scheduler=None,
        extra_modules=extras,
        device=torch.device("cpu"),
    )
    trainer.logger = Logger()
    config = _training_config(fim_measurements_per_epoch=2)

    trainer.epoch = 0
    trainer.run_epoch("train", config)
    trainer.epoch = 1
    trainer.run_epoch("train", config)

    assert fim.steps == [(0, "proper"), (3, "proper"), (6, "proper"), (9, "proper")]


def test_batchnorm_recalibration_updates_only_selected_buffers_and_restores_rng():
    from src.modules.batchnorm import recalibrate_batchnorm

    class BNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left_branch = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
            self.right_branch = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
            self.main_branch = torch.nn.Sequential(
                torch.nn.BatchNorm1d(4),
                torch.nn.Linear(4, 2),
            )

        def forward(
            self,
            left,
            right,
            enable_left_branch=True,
            enable_right_branch=True,
            **_kwargs,
        ):
            left = self.left_branch(left) if enable_left_branch else 0.0
            right = self.right_branch(right) if enable_right_branch else 0.0
            return self.main_branch(left + right)

    dataset = PairDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = BNModel().train()
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    left_mean_before = model.left_branch[0].running_mean.clone()
    right_mean_before = model.right_branch[0].running_mean.clone()
    main_mean_before = model.main_branch[0].running_mean.clone()
    rng_before = torch.get_rng_state().clone()

    report = recalibrate_batchnorm(
        model,
        loader,
        torch.device("cpu"),
        num_batches=1,
        scope="main_branch",
    )

    assert report["bn_recalibration/batches"] == 1
    assert report["bn_recalibration/modules"] == 1
    assert torch.equal(model.left_branch[0].running_mean, left_mean_before)
    assert torch.equal(model.right_branch[0].running_mean, right_mean_before)
    assert not torch.equal(model.main_branch[0].running_mean, main_mean_before)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name])
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert model.training is True


def test_phase4_batchnorm_control_runs_only_at_the_phase_boundary():
    trainer = object.__new__(TrainerClassification)
    trainer._initialize_run = lambda config: None
    trainer._apply_phase4_optimizer_overrides = lambda config: None
    calibrations = []
    trainer._maybe_recalibrate_phase4_batchnorm = (
        lambda config: calibrations.append(config.phase)
    )
    trainer._run_stage = lambda *_args, **_kwargs: None
    trainer.logger = type("Logger", (), {"close": lambda self: None})()
    config = SimpleNamespace(
        exp_starts_at_epoch=22,
        phase1_starts_at_epoch=0,
        phase1_ends_at_epoch=10,
        phase2_ends_at_epoch=20,
        phase3_ends_at_epoch=25,
        phase4_ends_at_epoch=40,
        phase4_target_train_acc=None,
    )

    trainer.run_all_at_once(config)
    assert calibrations == [4]

    calibrations.clear()
    config.exp_starts_at_epoch = 30
    trainer.run_all_at_once(config)
    assert calibrations == []

    config.exp_ends_at_epoch = 40
    trainer.run_phase(4, config)
    assert calibrations == []

    config.exp_starts_at_epoch = 25
    trainer.run_phase(4, config)
    assert calibrations == [4]

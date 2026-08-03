import gc

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from src.analysis.rsv import RSVProbeConfig, measure_rsv_layers
from src.modules.batchnorm import recalibrate_batchnorm
from src.modules.metrics import RunStatsBiModal
from src.utils.prepare import prepare_model
from src.utils.utils_model import load_model_specific_params
from src.utils.utils_trainer import (
    load_training_checkpoint,
    save_training_checkpoint,
)


pytestmark = pytest.mark.gpu


MODEL_CASES = [
    ("mm_mlp_bn", 1, 28, 14),
    ("mm_simple_cnn", 1, 28, 14),
    ("mm_resnet", 3, 64, 32),
    ("mm_effnetv2s", 3, 64, 32),
    ("mm_resnet18", 3, 64, 32),
    ("mm_convnext", 3, 64, 32),
]


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA compute node required")
    return torch.device("cuda")


@pytest.mark.parametrize("model_name,input_channels,height,width", MODEL_CASES)
def test_full_registered_model_forward_and_branch_intervention_on_gpu(
    model_name, input_channels, height, width
):
    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": input_channels,
        "img_height": height,
        "img_width": width,
        "overlap": 0.0,
        **load_model_specific_params(model_name),
    }
    model = prepare_model(model_name, model_params).to(device).eval()
    left = torch.randn(2, input_channels, height, width, device=device)
    right = torch.randn_like(left)

    with torch.no_grad():
        logits, left_features, right_features = model(left, right, return_features=True)
        intervention_logits = model(
            left,
            right,
            left_branch_intervention="deactivation",
            enable_left_branch=False,
        )

    assert logits.shape == intervention_logits.shape == (2, 5)
    assert left_features.device.type == right_features.device.type == "cuda"
    assert torch.isfinite(logits).all()
    assert torch.isfinite(intervention_logits).all()
    torch.cuda.synchronize()

    del model, left, right, logits, intervention_logits, left_features, right_features
    gc.collect()
    torch.cuda.empty_cache()


def test_full_resnet_training_step_produces_gradients_for_both_branches_on_gpu():
    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    left = torch.randn(4, 3, 64, 32, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1, 2, 3], device=device)

    loss = torch.nn.functional.cross_entropy(model(left, right), targets)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.left_branch.parameters())
    assert any(parameter.grad is not None for parameter in model.right_branch.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    torch.cuda.synchronize()


def test_phase3_disconnects_left_branch_and_freezes_bn_and_optimizer_state_on_gpu():
    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1
    )
    left = torch.randn(4, 3, 64, 32, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1, 2, 3], device=device)

    torch.nn.functional.cross_entropy(model(left, right), targets).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    left_parameters = list(model.left_branch.parameters())
    parameter_before = [parameter.detach().clone() for parameter in left_parameters]
    bn_before = {
        name: (module.running_mean.clone(), module.running_var.clone(), module.num_batches_tracked.clone())
        for name, module in model.left_branch.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    momentum_before = {
        id(parameter): optimizer.state[parameter]["momentum_buffer"].clone()
        for parameter in left_parameters
        if "momentum_buffer" in optimizer.state[parameter]
    }

    logits = model(
        left,
        right,
        left_branch_intervention="deactivation",
        enable_left_branch=False,
        enable_right_branch=True,
    )
    torch.nn.functional.cross_entropy(logits, targets).backward()

    assert all(parameter.requires_grad for parameter in left_parameters)
    assert all(parameter.grad is None for parameter in left_parameters)
    assert any(parameter.grad is not None for parameter in model.right_branch.parameters())
    assert any(parameter.grad is not None for parameter in model.main_branch.parameters())
    optimizer.step()

    for parameter, expected in zip(left_parameters, parameter_before):
        torch.testing.assert_close(parameter, expected)
        if id(parameter) in momentum_before:
            torch.testing.assert_close(
                optimizer.state[parameter]["momentum_buffer"], momentum_before[id(parameter)]
            )
    for name, module in model.left_branch.named_modules():
        if name in bn_before:
            mean, variance, batches = bn_before[name]
            torch.testing.assert_close(module.running_mean, mean)
            torch.testing.assert_close(module.running_var, variance)
            torch.testing.assert_close(module.num_batches_tracked, batches)


def test_relative_parity_phase3_updates_only_right_encoder_on_gpu():
    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device)
    model.requires_grad_(False)
    model.right_branch.requires_grad_(True)
    model.train()
    model.left_branch.eval()
    model.main_branch.eval()
    model.right_branch.train()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    frozen_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    frozen_bn = {
        name: (
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and not name.startswith("right_branch")
    }
    right_before = [
        parameter.detach().clone() for parameter in model.right_branch.parameters()
    ]
    left = torch.randn(4, 3, 64, 32, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1, 2, 3], device=device)
    logits = model(
        left,
        right,
        left_branch_intervention="deactivation",
        enable_left_branch=False,
        enable_right_branch=True,
    )
    torch.nn.functional.cross_entropy(logits, targets).backward()
    assert all(
        parameter.grad is None
        for parameter in model.left_branch.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.main_branch.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.right_branch.parameters()
    )
    optimizer.step()
    for name, expected in frozen_parameters.items():
        torch.testing.assert_close(model.get_parameter(name), expected)
    for name, module in model.named_modules():
        if name in frozen_bn:
            mean, variance, batches = frozen_bn[name]
            torch.testing.assert_close(module.running_mean, mean)
            torch.testing.assert_close(module.running_var, variance)
            torch.testing.assert_close(module.num_batches_tracked, batches)
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(model.right_branch.parameters(), right_before)
    )
    torch.cuda.synchronize()


def test_frozen_left_active_phase3_keeps_left_fixed_but_uses_its_features_on_gpu():
    from scripts.python_new.run_single import configure_phase_trainability

    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device)
    configure_phase_trainability(
        model,
        3,
        phase3_rule="local_accuracy",
        phase3_intervention="frozen_left_active",
    )
    model.train()
    model.left_branch.eval()
    model.right_branch.train()
    model.main_branch.train()
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )
    left_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.left_branch.named_parameters()
    }
    left_bn = {
        name: (
            module.running_mean.clone(),
            module.running_var.clone(),
            module.num_batches_tracked.clone(),
        )
        for name, module in model.left_branch.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }
    right_before = [
        parameter.detach().clone() for parameter in model.right_branch.parameters()
    ]
    main_before = [
        parameter.detach().clone() for parameter in model.main_branch.parameters()
    ]
    left = torch.randn(4, 3, 64, 32, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1, 2, 3], device=device)
    full_logits = model(
        left,
        right,
        enable_left_branch=True,
        enable_right_branch=True,
    )
    with torch.no_grad():
        weak_logits = model(
            left,
            right,
            left_branch_intervention="deactivation",
            enable_left_branch=False,
            enable_right_branch=True,
        )
    assert not torch.equal(full_logits.detach(), weak_logits)
    torch.nn.functional.cross_entropy(full_logits, targets).backward()
    assert all(parameter.grad is None for parameter in model.left_branch.parameters())
    assert any(parameter.grad is not None for parameter in model.right_branch.parameters())
    assert any(parameter.grad is not None for parameter in model.main_branch.parameters())
    optimizer.step()
    for name, expected in left_parameters.items():
        torch.testing.assert_close(model.left_branch.get_parameter(name), expected)
    for name, module in model.left_branch.named_modules():
        if name in left_bn:
            mean, variance, batches = left_bn[name]
            torch.testing.assert_close(module.running_mean, mean)
            torch.testing.assert_close(module.running_var, variance)
            torch.testing.assert_close(module.num_batches_tracked, batches)
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(model.right_branch.parameters(), right_before)
    )
    assert any(
        not torch.equal(parameter, expected)
        for parameter, expected in zip(model.main_branch.parameters(), main_before)
    )
    torch.cuda.synchronize()


def test_run_stats_supports_frozen_parity_groups_and_restores_modes_on_gpu():
    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device)
    initial_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    run_stats = RunStatsBiModal(model, initial_optimizer)
    model.requires_grad_(False)
    model.right_branch.requires_grad_(True)
    model.train()
    model.left_branch.eval()
    model.main_branch.eval()
    model.right_branch.train()
    expected_modes = {
        name: module.training for name, module in model.named_modules()
    }

    class CaptureLogger:
        def __init__(self):
            self.values = None

        def log_scalars(self, values, _step):
            self.values = values

    logger = CaptureLogger()
    run_stats.logger = logger
    run_stats("l2", 1)

    assert logger.values["run_stats_overall/left_branch_gradient_norm_squared"] == 0.0
    assert logger.values["run_stats_overall/main_branch_gradient_norm_squared"] == 0.0
    assert logger.values["run_stats_overall/left_branch_weight_norm_squared"] > 0.0
    assert logger.values["run_stats_overall/main_branch_weight_norm_squared"] > 0.0
    assert {
        name: module.training for name, module in model.named_modules()
    } == expected_modes


def test_unimodal_references_reproduce_full_bimodal_initialization_on_gpu():
    from scripts.python_new.run_single import _state_dict_sha256
    from src.utils.utils_trainer import manual_seed

    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    manual_seed(random_seed=83, device=device)
    source_model = prepare_model("mm_resnet", model_params).to(device)
    source_hash = _state_dict_sha256(source_model.state_dict())
    source_left_branch = {
        name: value.detach().clone()
        for name, value in source_model.left_branch.state_dict().items()
    }
    source_right_branch = {
        name: value.detach().clone()
        for name, value in source_model.right_branch.state_dict().items()
    }
    source_main_branch = {
        name: value.detach().clone()
        for name, value in source_model.main_branch.state_dict().items()
    }

    manual_seed(random_seed=83, device=device)
    left_reference = prepare_model("mm_resnet", model_params).to(device)
    left_reference_source_hash = _state_dict_sha256(left_reference.state_dict())

    manual_seed(random_seed=83, device=device)
    right_reference = prepare_model("mm_resnet", model_params).to(device)
    right_reference_source_hash = _state_dict_sha256(right_reference.state_dict())

    assert source_hash == left_reference_source_hash == right_reference_source_hash
    assert all(
        torch.equal(value, source_left_branch[name])
        for name, value in left_reference.left_branch.state_dict().items()
    )
    assert all(
        torch.equal(value, source_right_branch[name])
        for name, value in right_reference.right_branch.state_dict().items()
    )
    assert all(
        torch.equal(value, source_main_branch[name])
        for name, value in left_reference.main_branch.state_dict().items()
    )
    assert all(
        torch.equal(value, source_main_branch[name])
        for name, value in right_reference.main_branch.state_dict().items()
    )
    assert any(
        not torch.equal(value, source_left_branch[name])
        for name, value in source_right_branch.items()
    )




def test_full_umt_trainer_step_distills_only_active_branch_on_gpu():
    from collections import defaultdict
    from types import SimpleNamespace

    from src.modules.architectures.wrappers import BiModalModelwithPretrainedBranches
    from src.trainer.trainer_classification_mm_clp_umt import TrainerClassification

    device = require_cuda()
    model_params = {
        "num_classes": 5,
        "input_channels": 3,
        "img_height": 64,
        "img_width": 32,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    student = prepare_model("mm_resnet", model_params)
    teacher = prepare_model("mm_resnet", model_params)
    model = BiModalModelwithPretrainedBranches(
        student, teacher.left_branch, teacher.right_branch
    ).to(device).train()
    optimizer = torch.optim.SGD(model.main_model.parameters(), lr=0.01)

    class Criterion(torch.nn.Module):
        def forward(self, predictions, targets):
            loss = torch.nn.functional.cross_entropy(predictions, targets)
            return loss, {"loss": loss.item()}

    trainer = TrainerClassification(
        model=model,
        criterion=Criterion(),
        loaders={},
        optim=optimizer,
        lr_scheduler=None,
        extra_modules=defaultdict(lambda: None),
        device=device,
    )
    config = SimpleNamespace(
        extra={
            "left_branch_intervention": "deactivation",
            "right_branch_intervention": None,
            "enable_left_branch": False,
            "enable_right_branch": True,
        },
        distill=0.5,
    )
    left = torch.randn(2, 3, 64, 32, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1], device=device)

    loss, metrics = trainer.compute_loss(left, right, targets, config)
    loss.backward()

    assert "distillation/mse_left" not in metrics
    assert metrics["distillation/mse_right"] >= 0.0
    assert metrics["distillation/loss"] == metrics["distillation/mse_right"]
    assert metrics["distillation/weighted_loss"] == pytest.approx(
        0.5 * metrics["distillation/loss"]
    )
    assert metrics["loss"] == pytest.approx(loss.item())
    assert metrics["loss"] == pytest.approx(
        metrics["classification_loss"]
        + metrics["distillation/weighted_loss"]
    )
    assert any(parameter.grad is not None for parameter in student.right_branch.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in student.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    torch.cuda.synchronize()


def test_full_resnet_fim_probe_is_finite_and_rng_safe_on_gpu():
    from types import SimpleNamespace

    from src.modules.aux_modules import TraceFIM

    device = require_cuda()
    model_params = {
        "num_classes": 3,
        "input_channels": 3,
        "img_height": 32,
        "img_width": 16,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet18"),
    }
    model = prepare_model("mm_resnet18", model_params).to(device).train()
    held_out = {
        "proper_x_left": torch.randn(2, 3, 32, 16),
        "proper_x_right": torch.randn(2, 3, 32, 16),
        "blurred_x_right": torch.randn(2, 3, 32, 16),
    }
    config = SimpleNamespace(
        extra={
            "left_branch_intervention": None,
            "right_branch_intervention": None,
            "enable_left_branch": True,
            "enable_right_branch": True,
        }
    )
    probe = TraceFIM(
        held_out,
        model,
        num_classes=3,
        postfix="train",
        m_sampling=2,
        sampling_seed=71,
    )

    torch.cuda.manual_seed_all(123)
    state_before = torch.cuda.get_rng_state()
    metrics = probe(4, config, "proper")
    state_after = torch.cuda.get_rng_state()

    assert torch.equal(state_before, state_after)
    assert model.training is True
    assert metrics["trace_fim_overall_train/proper_trace1_weight"] > 0
    assert metrics["trace_fim_overall_train/proper_trace2_weight"] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_unimodal_resnet_fim_probe_accepts_one_trainable_branch_on_gpu():
    from types import SimpleNamespace

    from src.modules.aux_modules import TraceFIM

    device = require_cuda()
    model_params = {
        "num_classes": 3,
        "input_channels": 3,
        "img_height": 32,
        "img_width": 16,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet18"),
    }
    model = prepare_model("mm_resnet18", model_params).to(device).train()
    model.requires_grad_(False)
    model.left_branch.requires_grad_(True)
    held_out = {
        "proper_x_left": torch.randn(2, 3, 32, 16),
        "proper_x_right": torch.randn(2, 3, 32, 16),
        "blurred_x_right": torch.randn(2, 3, 32, 16),
    }
    config = SimpleNamespace(
        extra={
            "left_branch_intervention": None,
            "right_branch_intervention": "deactivation",
            "enable_left_branch": True,
            "enable_right_branch": False,
        }
    )
    metrics = TraceFIM(
        held_out,
        model,
        num_classes=3,
        postfix="train",
        m_sampling=1,
        sampling_seed=73,
    )(5, config, "proper")

    prefix = "trace_fim_overall_train"
    assert metrics[f"{prefix}/proper_trace1"] > 0
    assert metrics[f"{prefix}/proper_trace2"] == 0
    assert f"{prefix}/proper_trace1_per_parameter" in metrics
    assert f"{prefix}/proper_trace2_per_parameter" not in metrics
    assert f"{prefix}/proper_ratio_left_to_right" not in metrics


def test_full_model_training_checkpoint_round_trip_on_gpu(tmp_path):
    device = require_cuda()
    model_params = {
        "num_classes": 3,
        "input_channels": 3,
        "img_height": 32,
        "img_width": 16,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet18"),
    }
    model = prepare_model("mm_resnet18", model_params).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda _: 0.98
    )
    run_stats = RunStatsBiModal(model, optimizer)
    run_stats.start_phase(2)
    run_stats.model_trajectory_length_overall = 3.5
    left = torch.randn(2, 3, 32, 16, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1], device=device)
    torch.nn.functional.cross_entropy(model(left, right), targets).backward()
    optimizer.step()
    run_stats.record_optimizer_step()
    expected_trajectory_length = run_stats.model_trajectory_length_overall
    scheduler.step()

    checkpoint = tmp_path / "full-model.pth"
    save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint,
        next_epoch=5,
        global_step=20,
        diagnostics_state={"run_stats": run_stats.diagnostic_state_dict()},
    )
    restored = prepare_model("mm_resnet18", model_params).to(device)
    restored_optimizer = torch.optim.SGD(
        restored.parameters(), lr=0.5, momentum=0.9
    )
    restored_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        restored_optimizer, lr_lambda=lambda _: 1.0
    )
    state = load_training_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        restored_scheduler,
        device=device,
    )
    restored_run_stats = RunStatsBiModal(restored, restored_optimizer)
    restored_run_stats.load_diagnostic_state_dict(
        state["diagnostics_state"]["run_stats"]
    )

    assert state["next_epoch"] == 5
    assert state["global_step"] == 20
    assert restored_scheduler.lr_lambdas[0](0) == 0.98
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
    momentum = next(iter(restored_optimizer.state.values()))["momentum_buffer"]
    assert momentum.device.type == "cuda"
    assert restored_run_stats.current_phase == 2
    assert restored_run_stats.model_trajectory_length_overall == (
        expected_trajectory_length
    )
    assert next(restored_run_stats.phase_start_model.parameters()).is_cuda
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(
            restored_run_stats.optimizer_step_parameters[name], parameter
        )


def test_run_stats_handles_missing_gradients_on_gpu():
    device = require_cuda()
    model_params = {
        "num_classes": 3,
        "input_channels": 3,
        "img_height": 32,
        "img_width": 16,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet18"),
    }
    model = prepare_model("mm_resnet18", model_params).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_stats = RunStatsBiModal(model, optimizer)
    run_stats.logger = type(
        "CaptureLogger",
        (),
        {"log_scalars": lambda self, values, step: None},
    )()
    left = torch.randn(2, 3, 32, 16, device=device)
    right = torch.randn_like(left)
    targets = torch.tensor([0, 1], device=device)
    torch.nn.functional.cross_entropy(model(left, right), targets).backward()
    next(
        parameter
        for name, parameter in model.named_parameters()
        if name in run_stats.allowed_parameter_names and parameter.grad is not None
    ).grad = None

    run_stats("l2", 0)


def test_full_resnet_rsv_probe_on_gpu():
    device = require_cuda()

    class TinyRawDataset:
        classes = ("zero", "one")
        targets = [0, 0, 1, 1]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            generator = np.random.default_rng(index)
            image = generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            return Image.fromarray(image), self.targets[index]

    model_params = {
        "num_classes": 2,
        "input_channels": 3,
        "img_height": 32,
        "img_width": 16,
        "overlap": 0.0,
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params).to(device).train()
    results = measure_rsv_layers(
        model,
        TinyRawDataset(),
        ToTensor(),
        ToTensor(),
        config=RSVProbeConfig(
            samples_per_class=1,
            variants_per_source=3,
            translate_pixels=1,
            rotation_degrees=1,
            seed=4,
        ),
        batch_size=2,
        device=device,
    )

    assert model.training is True
    assert set(results) == {"stage3_avgpool", "stage4_avgpool"}
    for result in results.values():
        assert result["rsv"].shape[0] == 2
        assert torch.isfinite(result["rsv"]).all()
        assert result["rsv"].abs().max() <= 1
    assert results["stage3_avgpool"]["rsv"].shape[1] == 256
    assert results["stage4_avgpool"]["rsv"].shape[1] == 512

    class CalibrationDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            left = torch.full((3, 32, 16), float(index) / 4.0)
            right = torch.full((3, 32, 16), float(index + 1) / 4.0)
            return (left, right), index % 2

    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    report = recalibrate_batchnorm(
        model,
        DataLoader(CalibrationDataset(), batch_size=2, shuffle=True),
        device,
        num_batches=2,
        scope="main_branch",
    )
    assert report["bn_recalibration/batches"] == 2
    assert report["bn_recalibration/modules"] > 0
    assert report["bn_recalibration/running_mean_delta_l2"] > 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, parameters_before[name])

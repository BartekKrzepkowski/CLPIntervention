from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import (
    transforms_cifar10,
    transforms_fmnist,
    transforms_kmnist,
    transforms_mnist,
    transforms_svhn,
)
from src.data.probes import build_fim_probe
from src.data.normalization import normalization_from_transform
from src.modules.aux_modules import TraceFIM
from src.analysis.bootstrap import (
    PairedRSV,
    hierarchical_paired_bootstrap,
    paired_image_differences,
)
from src.analysis.rsv import (
    DEFAULT_RSV_LAYER_SPECS,
    RSVProbeConfig,
    affine_variants,
    balanced_fixed_count_indices,
    measure_rsv,
    measure_rsv_layers,
    save_rsv_result,
)
from src.modules.callbacks import RSVCallback, relative_source_variance
from src.modules.metrics import RunStatsBiModal


class TinyFIMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left_branch = torch.nn.Linear(2, 2, bias=False)
        self.right_branch = torch.nn.Linear(2, 2, bias=False)
        self.main_branch = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x_left, x_right, **kwargs):
        return self.main_branch(self.left_branch(x_left) + self.right_branch(x_right))


def test_trace_fim_is_deterministic_non_invasive_and_restores_model_mode():
    model = TinyFIMModel()
    model.train()
    held_out = {
        "proper_x_left": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "proper_x_right": torch.tensor([[0.5, 1.0], [1.0, 0.5]]),
        "blurred_x_right": torch.tensor([[0.25, 0.5], [0.5, 0.25]]),
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
        num_classes=2,
        postfix="train",
        m_sampling=3,
        sampling_seed=17,
    )

    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state()
    first = probe(10, config, "proper")
    state_after = torch.random.get_rng_state()
    second = probe(10, config, "proper")

    assert torch.equal(state_before, state_after)
    assert first == second
    assert model.training is True
    assert first["trace_fim_overall_train/proper_trace1_weight"] > 0
    assert first["trace_fim_overall_train/proper_trace2_weight"] > 0
    left_count = model.left_branch.weight.numel()
    right_count = model.right_branch.weight.numel()
    assert first["trace_fim_overall_train/proper_parameter_count1_weight"] == left_count
    assert first["trace_fim_overall_train/proper_parameter_count2_weight"] == right_count
    assert first["trace_fim_overall_train/proper_trace1_weight_per_parameter"] == pytest.approx(
        first["trace_fim_overall_train/proper_trace1_weight"] / left_count
    )
    assert first["trace_fim_overall_train/proper_trace2_weight_per_parameter"] == pytest.approx(
        first["trace_fim_overall_train/proper_trace2_weight"] / right_count
    )
    left_all_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "left_branch" in name and name in probe.penalized_parameter_names
    )
    right_all_count = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "right_branch" in name and name in probe.penalized_parameter_names
    )
    assert first["trace_fim_overall_train/proper_parameter_count1"] == left_all_count
    assert first["trace_fim_overall_train/proper_parameter_count2"] == right_all_count
    assert first[
        "trace_fim_overall_train/proper_ratio_left_to_right"
    ] == pytest.approx(
        first["trace_fim_overall_train/proper_trace1"]
        / first["trace_fim_overall_train/proper_trace2"]
    )
    assert first["trace_fim_overall_train/proper_trace1_per_parameter"] == pytest.approx(
        first["trace_fim_overall_train/proper_trace1"] / left_all_count
    )
    assert first["trace_fim_overall_train/proper_trace2_per_parameter"] == pytest.approx(
        first["trace_fim_overall_train/proper_trace2"] / right_all_count
    )



def test_trace_fim_matches_naive_per_example_sampled_fisher():
    model = TinyFIMModel().eval()
    held_out = {
        "proper_x_left": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "proper_x_right": torch.tensor([[0.5, 1.0], [1.0, 0.5]]),
        "blurred_x_right": torch.tensor([[0.25, 0.5], [0.5, 0.25]]),
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
        held_out, model, num_classes=2, postfix="train", m_sampling=2, sampling_seed=19
    )
    step = 7
    sampled_targets = probe._sample_targets(
        held_out["proper_x_left"], held_out["proper_x_right"], config, step
    )
    expected = 0.0
    for targets in sampled_targets:
        for index, target in enumerate(targets):
            logits = model(
                held_out["proper_x_left"][index:index + 1],
                held_out["proper_x_right"][index:index + 1],
            )
            loss = torch.nn.functional.cross_entropy(logits, target.unsqueeze(0))
            gradient = torch.autograd.grad(loss, model.left_branch.weight)[0]
            expected += gradient.square().sum().item()
    expected /= len(sampled_targets) * len(targets)

    measured = probe(step, config, "proper")

    assert measured["trace_fim_train_proper/left_branch.weight"] == pytest.approx(
        expected, rel=1e-6, abs=1e-8
    )





def test_trace_fim_chunking_preserves_the_estimator():
    model = TinyFIMModel().eval()
    held_out = {
        "proper_x_left": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        ),
        "proper_x_right": torch.tensor(
            [[0.5, 1.0], [1.0, 0.5], [0.25, 0.75]]
        ),
        "blurred_x_right": torch.tensor(
            [[0.25, 0.5], [0.5, 0.25], [0.125, 0.375]]
        ),
    }
    config = SimpleNamespace(
        extra={
            "left_branch_intervention": None,
            "right_branch_intervention": None,
            "enable_left_branch": True,
            "enable_right_branch": True,
        }
    )
    unchunked = TraceFIM(
        held_out, model, num_classes=2, postfix="train",
        m_sampling=3, sampling_seed=29, chunk_size=16,
    )(11, config, "proper")
    chunked = TraceFIM(
        held_out, model, num_classes=2, postfix="train",
        m_sampling=3, sampling_seed=29, chunk_size=1,
    )(11, config, "proper")

    assert chunked.keys() == unchunked.keys()
    for key in chunked:
        assert chunked[key] == pytest.approx(
            unchunked[key], rel=1e-6, abs=1e-8
        )

def test_run_stats_uses_exact_optimizer_displacement_with_decay_and_momentum():
    model = TinyFIMModel()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(float(index))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.5
    )
    run_stats = RunStatsBiModal(model, optimizer)
    expected_group_lengths = {
        "left": 0.0,
        "right": 0.0,
        "main": 0.0,
    }
    expected_model_length = 0.0

    for _ in range(2):
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        step_squared = {"left": 0.0, "right": 0.0, "main": 0.0}
        for name, parameter in model.named_parameters():
            displacement = torch.linalg.vector_norm(
                parameter.detach() - before[name]
            ).item()
            group = (
                "left" if "left_branch" in name
                else "right" if "right_branch" in name
                else "main"
            )
            step_squared[group] += displacement ** 2
        step_lengths = {
            group: squared ** 0.5 for group, squared in step_squared.items()
        }
        for group, length in step_lengths.items():
            expected_group_lengths[group] += length
        expected_model_length += sum(
            length ** 2 for length in step_lengths.values()
        ) ** 0.5
        run_stats.record_optimizer_step()

    assert run_stats.left_branch_trajectory_length_overall == pytest.approx(
        expected_group_lengths["left"]
    )
    assert run_stats.right_branch_trajectory_length_overall == pytest.approx(
        expected_group_lengths["right"]
    )
    assert run_stats.main_branch_trajectory_length_overall == pytest.approx(
        expected_group_lengths["main"]
    )
    assert run_stats.model_trajectory_length_overall == pytest.approx(
        expected_model_length
    )
    assert run_stats.model_trajectory_length_overall > 0.0


def test_run_stats_rejects_unimplemented_distance_types():
    model = TinyFIMModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_stats = RunStatsBiModal(model, optimizer)

    with pytest.raises(ValueError, match="use 'l2'"):
        run_stats.distance_between_models(
            model,
            model,
            defaultdict(float),
            distance_type="cosine",
            dist_label="test",
        )


class CaptureLogger:
    def __init__(self):
        self.values = []

    def log_scalars(self, values, step):
        self.values.append((values, step))


def test_run_stats_logs_separate_branch_distances_from_each_phase_start():
    model = TinyFIMModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_stats = RunStatsBiModal(model, optimizer)
    run_stats.logger = CaptureLogger()
    run_stats.start_phase(1)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.add_(float(index))
    run_stats("l2", 4)
    values = run_stats.logger.values[-1][0]
    for branch in ("left_branch", "right_branch", "main_branch"):
        key = f"run_stats_overall/{branch}_distance_from_phase1_start_l2"
        assert values[key] > 0.0
    run_stats.start_phase(2)
    run_stats("l2", 5)
    values = run_stats.logger.values[-1][0]
    for branch in ("left_branch", "right_branch", "main_branch"):
        key = f"run_stats_overall/{branch}_distance_from_phase2_start_l2"
        assert values[key] == pytest.approx(0.0)

def test_run_stats_diagnostic_state_round_trip_preserves_references_and_totals():
    model = TinyFIMModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_stats = RunStatsBiModal(model, optimizer)
    run_stats.start_phase(3)
    run_stats.model_trajectory_length_overall = 12.5
    parameter_name = next(iter(run_stats.left_branch_trajectory_length_group))
    run_stats.left_branch_trajectory_length_group[parameter_name] = 2.25
    state = run_stats.diagnostic_state_dict()
    assert "optimizer_step_parameters" not in state
    restored = RunStatsBiModal(model, optimizer)
    restored.load_diagnostic_state_dict(state)
    assert restored.current_phase == 3
    assert restored.model_trajectory_length_overall == 12.5
    assert (
        restored.left_branch_trajectory_length_group[parameter_name] == 2.25
    )
    for reference_name in (
        "model_zero",
        "last_model",
        "phase_start_model",
    ):
        expected = getattr(run_stats, reference_name).state_dict()
        actual = getattr(restored, reference_name).state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
    for name, expected in run_stats.optimizer_step_parameters.items():
        torch.testing.assert_close(
            restored.optimizer_step_parameters[name], expected
        )


class TinyImageDataset:
    classes = ("zero", "one")

    def __init__(self):
        self.targets = [0] * 10 + [1] * 10

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        pixels = np.full((8, 8, 3), index * 10, dtype=np.uint8)
        return Image.fromarray(pixels), self.targets[index]


def test_generated_fim_probe_is_balanced_paired_and_disjoint():
    dataset = TinyImageDataset()
    first = build_fim_probe(
        dataset, "mm_cifar10", resize_factor=0.25, fraction=0.2, seed=9
    )
    second = build_fim_probe(
        dataset, "mm_cifar10", resize_factor=0.25, fraction=0.2, seed=9
    )

    selected_labels = np.asarray(dataset.targets)[first.probe_indices]
    assert np.bincount(selected_labels).tolist() == [2, 2]
    assert not np.intersect1d(first.probe_indices, first.train_indices).size
    assert np.array_equal(
        np.sort(np.concatenate((first.probe_indices, first.train_indices))),
        np.arange(len(dataset)),
    )
    assert torch.equal(first.tensors["y"], torch.as_tensor(selected_labels))
    assert np.array_equal(first.probe_indices, second.probe_indices)
    for name in first.tensors:
        assert torch.equal(first.tensors[name], second.tensors[name])


def test_rsv_uses_kleinman_plus_left_minus_right_convention_and_handles_zero_variance():
    left_varying = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    right_fixed = torch.zeros_like(left_varying)
    assert torch.equal(
        relative_source_variance(left_varying, right_fixed),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(
        relative_source_variance(right_fixed, left_varying),
        torch.tensor([-1.0, 0.0]),
    )

    callback = RSVCallback(group_size=4)
    callback.enable()
    callback(
        torch.nn.Linear(1, 1),
        (),
        torch.tensor([[-1.0], [1.0], [0.0], [0.0]]),
    )
    assert torch.equal(callback.gather_data()[0][0], torch.tensor([1.0]))


def test_rsv_probe_is_balanced_deterministic_raw_and_manifested(tmp_path):
    class ProbeDataset(TinyImageDataset):
        def __init__(self):
            self.targets = [0] * 5 + [1] * 5

        def __getitem__(self, index):
            grid = np.arange(64, dtype=np.uint8).reshape(8, 8)
            pixels = np.stack((grid, np.roll(grid, index, axis=1), grid), axis=2)
            return Image.fromarray(pixels), self.targets[index]

    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.main_branch = torch.nn.Sequential(torch.nn.Linear(192, 4, bias=False))

        def forward(self, left, right):
            return self.main_branch(torch.cat((left.flatten(1), right.flatten(1)), dim=1))

    from torchvision.transforms import ToTensor

    dataset = ProbeDataset()
    config = RSVProbeConfig(
        samples_per_class=2,
        variants_per_source=4,
        translate_pixels=1,
        rotation_degrees=2,
        seed=7,
    )
    model = ProbeModel().train()
    first = measure_rsv(
        model, dataset, ToTensor(), ToTensor(), config=config, batch_size=2
    )
    second = measure_rsv(
        model, dataset, ToTensor(), ToTensor(), config=config, batch_size=3
    )

    assert model.training is True
    assert first["rsv"].shape == (4, 4)
    torch.testing.assert_close(first["rsv"], second["rsv"], rtol=1e-5, atol=1e-7)
    assert torch.equal(first["selected_indices"], second["selected_indices"])
    assert first["rsv"].abs().max() <= 1
    assert first["metadata"]["layer"] == "main_branch.0"
    assert first["metadata"]["inference_batch_size"] == 2
    assert first["metadata"]["sign_convention"] == "+1=left,-1=right (Kleinman et al.)"

    raw_path = tmp_path / "rsv.pt"
    manifest_path = save_rsv_result(first, raw_path)
    loaded = torch.load(raw_path, weights_only=True)
    assert torch.equal(loaded["rsv"], first["rsv"])
    assert '"value_count": 16' in manifest_path.read_text(encoding="utf-8")


def test_rsv_selection_and_variants_include_original():
    indices = balanced_fixed_count_indices([0, 0, 0, 1, 1, 1], 2, seed=2)
    assert np.bincount(np.asarray([0, 0, 0, 1, 1, 1])[indices]).tolist() == [2, 2]

    field = Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4))
    config = RSVProbeConfig(variants_per_source=3, seed=2)
    variants = affine_variants(field, config, torch.Generator().manual_seed(2))
    assert np.array_equal(np.asarray(variants[0]), np.asarray(field))
    assert len(variants) == 3


def test_paper_training_transforms_shift_but_do_not_rotate():
    transforms_to_check = (
        transforms_cifar10.transform_train_proper(0.0, "left"),
        transforms_cifar10.transform_train_blurred(32, 32, 0.25, 0.0),
        transforms_fmnist.transform_train_proper(0.0, "left"),
        transforms_fmnist.transform_train_blurred(28, 28, 0.25, 0.0),
    )
    for pipeline in transforms_to_check:
        affine = next(
            transform
            for transform in pipeline.transforms
            if type(transform).__name__ == "RandomAffine"
        )
        assert affine.degrees == [0.0, 0.0]
        assert affine.translate == (1 / 8, 1 / 8)


@pytest.mark.parametrize(
    "module", (transforms_mnist, transforms_kmnist, transforms_svhn)
)
def test_unverified_normalization_fails_closed(module):
    with pytest.raises(ValueError, match="unavailable or unverified"):
        module.transform_eval_proper(0.0, "left")


def test_normalization_metadata_records_exact_transform_values():
    transform = transforms_cifar10.transform_eval_blurred(32, 32, 0.25, 0.0)
    metadata = normalization_from_transform(transform)

    assert metadata["mean"] == list(
        transforms_cifar10.OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[0.0][0]
    )
    assert metadata["std"] == list(
        transforms_cifar10.OVERLAP_TO_NORMALIZATION_MAP_BLURRED_R[0.0][1]
    )
def test_dual_rsv_protocol_pools_stage3_and_stage4_in_shared_forwards():
    from torchvision.transforms import ToTensor

    class ProbeDataset:
        targets = [0, 0, 1, 1]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            generator = np.random.default_rng(index)
            pixels = generator.integers(0, 256, (8, 8, 3), dtype=np.uint8)
            return Image.fromarray(pixels), self.targets[index]

    class ProbeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.main_branch = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1),
                torch.nn.Conv2d(4, 5, 1),
            )
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(5, 2)
            self.forward_calls = 0

        def forward(self, left, right):
            self.forward_calls += 1
            features = self.main_branch(left + right)
            return self.fc(self.avgpool(features).flatten(1))

    model = ProbeModel().train()
    results = measure_rsv_layers(
        model,
        ProbeDataset(),
        ToTensor(),
        ToTensor(),
        config=RSVProbeConfig(
            samples_per_class=1,
            variants_per_source=3,
            translate_pixels=1,
            rotation_degrees=1,
            seed=4,
        ),
        layer_specs=DEFAULT_RSV_LAYER_SPECS,
        batch_size=2,
    )

    assert set(results) == {"stage3_avgpool", "stage4_avgpool"}
    assert results["stage3_avgpool"]["rsv"].shape == (2, 4)
    assert results["stage4_avgpool"]["rsv"].shape == (2, 5)
    assert results["stage3_avgpool"]["metadata"]["layer"] == "main_branch.0"
    assert results["stage3_avgpool"]["metadata"]["spatial_average_pool"] is True
    assert results["stage4_avgpool"]["metadata"]["layer"] == "avgpool"
    assert results["stage4_avgpool"]["metadata"]["spatial_average_pool"] is False
    assert model.forward_calls == 8
    assert model.training is True


def test_hierarchical_bootstrap_uses_paired_model_and_image_differences():
    def artifact(value):
        return {
            "format": "clpintervention.rsv",
            "version": 3,
            "rsv": torch.full((4, 3), value),
            "selected_indices": torch.tensor([2, 4, 6, 8]),
            "selected_labels": torch.tensor([0, 0, 1, 1]),
            "metadata": {
                "measurement": "stage4_avgpool",
                "layer": "avgpool",
                "spatial_average_pool": False,
                "sign_convention": "+1=left,-1=right (Kleinman et al.)",
            },
        }

    pairs = [
        PairedRSV("seed1", artifact(0.0), artifact(0.25)),
        PairedRSV("seed2", artifact(-0.1), artifact(0.15)),
    ]
    result = hierarchical_paired_bootstrap(pairs, replicates=100, seed=3)

    assert result["difference"] == "intervention-control"
    assert result["observed"] == pytest.approx(0.25)
    assert result["confidence_interval"] == pytest.approx([0.25, 0.25])
    assert result["bootstrap_probability_above_zero"] == 1.0
    assert result["warning"] is not None

    mismatched = artifact(0.25)
    mismatched["selected_indices"] = torch.tensor([1, 4, 6, 8])
    with pytest.raises(ValueError, match="mismatched selected_indices"):
        paired_image_differences(
            PairedRSV("bad", artifact(0.0), mismatched)
        )

    other_control = artifact(0.0)
    other_intervention = artifact(0.1)
    for item in (other_control, other_intervention):
        item["metadata"]["measurement"] = "stage3_avgpool"
        item["metadata"]["layer"] = "main_branch.0"
        item["metadata"]["spatial_average_pool"] = True
    with pytest.raises(ValueError, match="protocol differs between models"):
        hierarchical_paired_bootstrap(
            (pairs[0], PairedRSV("seed3", other_control, other_intervention))
        )

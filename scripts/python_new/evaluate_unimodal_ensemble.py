"""Post-hoc evaluation of paired clean unimodal references and their ensemble.

This module is intentionally evaluation-only.  The primary UME definition is
equal-weight soft voting (arithmetic mean of posterior probabilities).  An
equal-weight logit mean is reported as a sensitivity analysis and is never
selected on validation data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.trainer.modality_evaluation import (
    DOMINANT_ONLY_MODE,
    FULL_MODE,
    WEAK_ONLY_MODE,
)
from src.trainer.unimodal_references import (
    UnimodalReferenceMetadata,
    validate_unimodal_reference_pair,
)
from src.utils.prepare import prepare_model
from src.utils.prepare_clp_data import (
    prepare_test_loaders_clp,
    prepare_training_loaders_clp,
)
from src.utils.utils_model import load_model_specific_params
from src.utils.utils_trainer import (
    load_checkpoint_metadata,
    load_training_checkpoint,
)


@dataclass
class ClassificationAccumulator:
    """Streaming proper-scoring and calibration metrics from probabilities."""

    num_bins: int = 15
    count: int = 0
    correct: int = 0
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    confidence_sum: float = 0.0
    incorrect_confidence_sum: float = 0.0
    incorrect_count: int = 0
    bin_count: list[int] = field(default_factory=list)
    bin_confidence_sum: list[float] = field(default_factory=list)
    bin_correct_sum: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.bin_count:
            self.bin_count = [0] * self.num_bins
            self.bin_confidence_sum = [0.0] * self.num_bins
            self.bin_correct_sum = [0] * self.num_bins

    def update(self, probabilities: torch.Tensor, targets: torch.Tensor):
        probabilities = probabilities.detach()
        targets = targets.detach()
        batch_count = int(targets.numel())
        predicted = probabilities.argmax(dim=1)
        correctness = predicted.eq(targets)
        confidence = probabilities.max(dim=1).values
        target_probabilities = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        one_hot = F.one_hot(
            targets, num_classes=probabilities.shape[1]
        ).to(probabilities.dtype)

        self.count += batch_count
        self.correct += int(correctness.sum())
        self.nll_sum += float(
            -target_probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
            .log()
            .sum()
        )
        self.brier_sum += float(((probabilities - one_hot) ** 2).sum())
        self.confidence_sum += float(confidence.sum())
        incorrect = ~correctness
        self.incorrect_count += int(incorrect.sum())
        self.incorrect_confidence_sum += float(confidence[incorrect].sum())

        bin_indices = torch.clamp(
            (confidence * self.num_bins).to(torch.long),
            max=self.num_bins - 1,
        )
        for index in range(self.num_bins):
            in_bin = bin_indices.eq(index)
            if not bool(in_bin.any()):
                continue
            self.bin_count[index] += int(in_bin.sum())
            self.bin_confidence_sum[index] += float(confidence[in_bin].sum())
            self.bin_correct_sum[index] += int(correctness[in_bin].sum())

    def result(self):
        if not self.count:
            raise ValueError("evaluation loader is empty")
        ece = 0.0
        for count, confidence_sum, correct_sum in zip(
            self.bin_count,
            self.bin_confidence_sum,
            self.bin_correct_sum,
        ):
            if count:
                ece += (count / self.count) * abs(
                    confidence_sum / count - correct_sum / count
                )
        return {
            "samples": self.count,
            "accuracy": self.correct / self.count,
            "nll": self.nll_sum / self.count,
            "brier": self.brier_sum / self.count,
            "ece_15": ece,
            "mean_confidence": self.confidence_sum / self.count,
            "mean_incorrect_confidence": (
                self.incorrect_confidence_sum / self.incorrect_count
                if self.incorrect_count
                else None
            ),
        }


def ensemble_probabilities(left_logits, right_logits):
    """Return pre-registered probability-mean UME and logit sensitivity."""
    left_probabilities = torch.softmax(left_logits, dim=1)
    right_probabilities = torch.softmax(right_logits, dim=1)
    return {
        "unimodal_left": left_probabilities,
        "unimodal_right": right_probabilities,
        "ume_probability_mean": 0.5 * (
            left_probabilities + right_probabilities
        ),
        "ume_logit_mean_sensitivity": torch.softmax(
            0.5 * (left_logits + right_logits), dim=1
        ),
    }


def ensemble_with_gold_probabilities(left_logits, right_logits, gold_logits):
    """Equal-weight three-model ensemble and its logit sensitivity."""
    left_probabilities = torch.softmax(left_logits, dim=1)
    right_probabilities = torch.softmax(right_logits, dim=1)
    gold_probabilities = torch.softmax(gold_logits, dim=1)
    return {
        "ume_plus_gold_probability_mean": (
            left_probabilities + right_probabilities + gold_probabilities
        )
        / 3.0,
        "ume_plus_gold_logit_mean_sensitivity": torch.softmax(
            (left_logits + right_logits + gold_logits) / 3.0, dim=1
        ),
    }


def _load_model(path, model_params, device):
    model = prepare_model("mm_resnet", model_params=model_params).to(device)
    load_training_checkpoint(path, model, device=device)
    model.eval()
    return model


def _validate_gold_checkpoint(path, *, seed, reference):
    metadata = load_checkpoint_metadata(path)["metadata"]
    protocol = metadata.get("protocol_manifest")
    if not protocol:
        raise ValueError("gold checkpoint lacks protocol_manifest")
    expected = {
        "seed": int(seed),
        "model": reference.model_name,
        "dataset": reference.dataset_name,
        "split_profile": reference.split_profile,
        "normalization_profile": reference.normalization_profile,
    }
    observed = {
        "seed": int(protocol["training"]["seed"]),
        "model": str(protocol["model"]["name"]),
        "dataset": str(protocol["dataset"]["name"]),
        "split_profile": str(protocol["dataset"]["split_profile"]),
        "normalization_profile": str(
            protocol["dataset"]["normalization_profile"]
        ),
    }
    if observed != expected:
        raise ValueError(
            f"gold checkpoint protocol mismatch: {observed} != {expected}"
        )
    if protocol["dataset"]["split"] != reference.split_manifest:
        raise ValueError("gold checkpoint uses a different dataset split")


def _evaluate_loader(left_model, right_model, gold_model, loader, device):
    names = (
        "unimodal_left",
        "unimodal_right",
        "ume_probability_mean",
        "ume_logit_mean_sensitivity",
        "ume_plus_gold_probability_mean",
        "ume_plus_gold_logit_mean_sensitivity",
        "gold_full",
        "gold_dominant_only",
        "gold_weak_only",
    )
    accumulators = {name: ClassificationAccumulator() for name in names}
    with torch.no_grad():
        for (x_left, x_right), targets in loader:
            x_left = x_left.to(device, non_blocking=True)
            x_right = x_right.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            left_logits = left_model(
                x_left, x_right, **DOMINANT_ONLY_MODE.kwargs()
            )
            right_logits = right_model(
                x_left, x_right, **WEAK_ONLY_MODE.kwargs()
            )
            probabilities = ensemble_probabilities(left_logits, right_logits)
            gold_full_logits = gold_model(
                x_left, x_right, **FULL_MODE.kwargs()
            )
            gold_full_probabilities = torch.softmax(gold_full_logits, dim=1)
            probabilities.update(
                ensemble_with_gold_probabilities(
                    left_logits, right_logits, gold_full_logits
                )
            )
            probabilities.update(
                {
                    "gold_full": gold_full_probabilities,
                    "gold_dominant_only": torch.softmax(
                        gold_model(
                            x_left, x_right, **DOMINANT_ONLY_MODE.kwargs()
                        ),
                        dim=1,
                    ),
                    "gold_weak_only": torch.softmax(
                        gold_model(x_left, x_right, **WEAK_ONLY_MODE.kwargs()),
                        dim=1,
                    ),
                }
            )
            for name, values in probabilities.items():
                accumulators[name].update(values, targets)
    return {name: accumulator.result() for name, accumulator in accumulators.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--left-checkpoint", required=True)
    parser.add_argument("--right-checkpoint", required=True)
    parser.add_argument("--gold-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/cifar10_relative_unimodal_parity_p1_40.yaml",
    )
    parser.add_argument("--wandb-project", default="CLPIntervention_UnimodalParity")
    parser.add_argument("--wandb-entity", default="bartekk")
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("UME dataset evaluation must run on a GPU compute node")
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)
    left_reference = UnimodalReferenceMetadata.from_checkpoint(
        args.left_checkpoint
    )
    right_reference = UnimodalReferenceMetadata.from_checkpoint(
        args.right_checkpoint
    )
    validate_unimodal_reference_pair(
        left_reference,
        right_reference,
        seed=args.seed,
        model_name="mm_resnet",
        dataset_name="mm_cifar10",
        split_profile=str(config.split_profile),
        normalization_profile=str(config.normalization_profile),
        split_manifest=left_reference.split_manifest,
        normalization_manifest=left_reference.normalization_manifest,
    )
    _validate_gold_checkpoint(
        args.gold_checkpoint, seed=args.seed, reference=left_reference
    )

    dataset_params = {
        "dataset_path": None,
        "overlap": float(config.overlap),
        "resize_factor": float(config.resize_factor),
        "subset": None,
    }
    loader_params = {
        "batch_size": int(config.batch_size),
        "pin_memory": True,
        "num_workers": int(config.num_workers),
    }
    training_loaders = prepare_training_loaders_clp(
        "mm_cifar10",
        dataset_params,
        loader_params,
        split_profile=str(config.split_profile),
        normalization_profile=str(config.normalization_profile),
        generator_seed=args.seed,
        verify_dataset_files=bool(config.verify_dataset_files),
    )
    if training_loaders.split_manifest != left_reference.split_manifest:
        raise ValueError("runtime validation split differs from references")
    test_loaders = prepare_test_loaders_clp(
        "mm_cifar10",
        dataset_params,
        loader_params,
        normalization_profile=str(config.normalization_profile),
    )

    sample_inputs, _ = training_loaders.validation_proper.dataset[0]
    input_channels, img_height, img_width = sample_inputs[0].shape
    model_params = {
        "num_classes": 10,
        "input_channels": input_channels,
        "img_height": img_height,
        "img_width": img_width,
        "overlap": float(config.overlap),
        **load_model_specific_params("mm_resnet"),
    }
    left_model = _load_model(args.left_checkpoint, model_params, device)
    right_model = _load_model(args.right_checkpoint, model_params, device)
    gold_model = _load_model(args.gold_checkpoint, model_params, device)

    results = {
        "definition": {
            "primary": "equal_weight_arithmetic_mean_of_softmax_probabilities",
            "sensitivity": "equal_weight_arithmetic_mean_of_logits",
            "plus_gold_primary": (
                "equal_weight_arithmetic_mean_of_left_right_and_gold_"
                "softmax_probabilities"
            ),
            "plus_gold_sensitivity": (
                "equal_weight_arithmetic_mean_of_left_right_and_gold_logits"
            ),
            "weights_selected_on_validation": False,
            "test_policy": "posthoc_after_fixed_ensemble_definition",
        },
        "seed": args.seed,
        "checkpoints": {
            "left": args.left_checkpoint,
            "right": args.right_checkpoint,
            "gold": args.gold_checkpoint,
        },
        "reference_validation_accuracy_from_checkpoint": {
            "left": left_reference.validation_accuracy,
            "right": right_reference.validation_accuracy,
        },
        "validation_proper": _evaluate_loader(
            left_model,
            right_model,
            gold_model,
            training_loaders.validation_proper,
            device,
        ),
        "test_proper": _evaluate_loader(
            left_model,
            right_model,
            gold_model,
            test_loaders.test_proper,
            device,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb_mode,
            name=f"posthoc-ume-plus-gold-seed-{args.seed}",
            job_type="posthoc_unimodal_ensemble",
            config=results["definition"] | {"seed": args.seed},
        )
        flattened = {}
        for split in ("validation_proper", "test_proper"):
            for model_name, metrics in results[split].items():
                for metric_name, value in metrics.items():
                    if value is not None:
                        flattened[f"posthoc_ume/{split}/{model_name}/{metric_name}"] = value
        run.log(flattened)
        run.summary.update(flattened)
        run.finish()
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()

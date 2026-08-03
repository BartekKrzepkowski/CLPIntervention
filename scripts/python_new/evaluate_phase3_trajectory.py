"""Create validation/test modality trajectories from retained P3 checkpoints.

Validation values are read from the immutable online trajectory.  Test proper
is evaluated only post hoc at pre-registered retained milestones and is never
passed to a phase controller or checkpoint selector.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.trainer.modality_evaluation import WEAK_ONLY_MODE, evaluate_modalities
from src.utils.prepare import prepare_model
from src.utils.prepare_clp_data import prepare_test_loaders_clp
from src.utils.utils_model import load_model_specific_params
from src.utils.utils_trainer import load_training_checkpoint


MODES = ("full", "dominant_only", "weak_only")
COLORS = {
    "full": "#2563eb",
    "dominant_only": "#dc2626",
    "weak_only": "#16a34a",
}


def _compact_metrics(metrics):
    """Keep aggregates; this plot artifact does not need per-example arrays."""
    state = metrics.state_dict()
    return {
        "phase_epoch": state["phase_epoch"],
        "global_epoch": state["global_epoch"],
        "global_step": state["global_step"],
        **{mode: state[mode] for mode in (*MODES, "intervention")},
    }


def _parse_trajectory(value):
    seed, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("trajectory must use SEED=PATH")
    return int(seed), Path(path)


def _load_records(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"empty trajectory: {path}")
    return records


def _metric_row(seed, split, record, metrics=None):
    source = record["metrics"] if metrics is None else metrics.state_dict()
    return {
        "seed": int(seed),
        "split": split,
        "phase_epoch": int(record["phase_epoch"]),
        **{
            f"{mode}_accuracy": float(source[mode]["accuracy"])
            for mode in MODES
        },
    }


def _build_model_and_test_loader(config, device):
    dataset_params = {
        "dataset_path": None,
        "overlap": float(config["overlap"]),
        "resize_factor": float(config["resize_factor"]),
        "subset": None,
    }
    loader_params = {
        "batch_size": int(config["batch_size"]),
        "pin_memory": True,
        "num_workers": int(config["num_workers"]),
    }
    loaders = prepare_test_loaders_clp(
        "mm_cifar10",
        dataset_params,
        loader_params,
        normalization_profile=str(config["normalization_profile"]),
    )
    sample, _ = loaders.test_proper.dataset[0]
    channels, height, width = sample[0].shape
    model_params = {
        "num_classes": 10,
        "input_channels": channels,
        "img_height": height,
        "img_width": width,
        "overlap": float(config["overlap"]),
        **load_model_specific_params("mm_resnet"),
    }
    model = prepare_model("mm_resnet", model_params=model_params).to(device)
    return model, loaders.test_proper


def _posthoc_test_rows(seed, records, model, loader, device, interval):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    rows = []
    sidecar = []
    for record in records:
        epoch = int(record["phase_epoch"])
        if epoch % interval or not record.get("checkpoint_retained", False):
            continue
        checkpoint = record.get("checkpoint_path")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"retained checkpoint missing at e3={epoch}: {checkpoint}"
            )
        load_training_checkpoint(checkpoint, model, device=device)
        metrics = evaluate_modalities(
            model,
            criterion,
            loader,
            device,
            intervention_mode=WEAK_ONLY_MODE,
            phase_epoch=epoch,
            global_epoch=int(record["global_epoch"]),
            global_step=int(record["global_step"]),
        )
        row = _metric_row(seed, "test_proper_posthoc", record, metrics)
        rows.append(row)
        sidecar.append(
            {
                "version": 1,
                "seed": seed,
                "phase_epoch": epoch,
                "checkpoint_path": checkpoint,
                "test_used_for_decision": False,
                "metrics": _compact_metrics(metrics),
            }
        )
    return rows, sidecar


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, split, path, title):
    selected = [row for row in rows if row["split"] == split]
    by_seed = defaultdict(list)
    for row in selected:
        by_seed[row["seed"]].append(row)
    fig, axis = plt.subplots(figsize=(10, 6))
    for mode in MODES:
        key = f"{mode}_accuracy"
        for seed_rows in by_seed.values():
            seed_rows.sort(key=lambda row: row["phase_epoch"])
            axis.plot(
                [row["phase_epoch"] for row in seed_rows],
                [row[key] for row in seed_rows],
                color=COLORS[mode],
                alpha=0.18,
                linewidth=1.0,
            )
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["phase_epoch"]].append(row[key])
        epochs = sorted(grouped)
        averages = [mean(grouped[epoch]) for epoch in epochs]
        deviations = [
            stdev(grouped[epoch]) if len(grouped[epoch]) > 1 else 0.0
            for epoch in epochs
        ]
        axis.plot(
            epochs,
            averages,
            color=COLORS[mode],
            linewidth=2.5,
            marker="o" if split.startswith("test") else None,
            markersize=4,
            label=mode.replace("_", "-"),
        )
        axis.fill_between(
            epochs,
            [value - delta for value, delta in zip(averages, deviations)],
            [value + delta for value, delta in zip(averages, deviations)],
            color=COLORS[mode],
            alpha=0.10,
        )
    axis.set(xlabel="Phase 3 epoch", ylabel="Accuracy", title=title)
    axis.set_xlim(left=0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", action="append", type=_parse_trajectory, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-interval", type=int, default=20)
    parser.add_argument("--label", default="P1=120, P2=200, Phase 3")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("trajectory dataset evaluation requires a GPU compute node")
    if args.test_interval <= 0:
        raise ValueError("test interval must be positive")
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, test_loader = _build_model_and_test_loader(config, device)
    rows = []
    for seed, path in args.trajectory:
        records = _load_records(path)
        rows.extend(
            _metric_row(seed, "validation_proper", record)
            for record in records
        )
        test_rows, sidecar = _posthoc_test_rows(
            seed, records, model, test_loader, device, args.test_interval
        )
        rows.extend(test_rows)
        sidecar_path = args.output_dir / f"phase3_test_trajectory_seed{seed}.jsonl"
        sidecar_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in sidecar),
            encoding="utf-8",
        )
    _write_csv(args.output_dir / "phase3_validation_and_test_accuracy.csv", rows)
    _plot(
        rows,
        "validation_proper",
        args.output_dir / "phase3_validation_accuracy.png",
        f"{args.label} — validation proper",
    )
    _plot(
        rows,
        "test_proper_posthoc",
        args.output_dir / "phase3_test_accuracy.png",
        f"{args.label} — test proper (post hoc, not used for decisions)",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows)}))


if __name__ == "__main__":
    main()

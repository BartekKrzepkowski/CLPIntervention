"""Replay recovery-fraction Phase-3 stopping on saved trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.trainer.modality_evaluation import ModalityEvaluationResult
from src.trainer.validation_control import (
    Phase3RelativeUnimodalStopper,
    Phase3StopConfig,
)


def load_trajectory(path):
    by_epoch = {}
    with Path(path).open(encoding="utf-8") as trajectory_file:
        for line in trajectory_file:
            record = json.loads(line)
            if int(record.get("version", 0)) != 1:
                raise ValueError(f"unsupported trajectory version in {path}")
            by_epoch[int(record["phase_epoch"])] = record
    records = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if not records or int(records[0]["phase_epoch"]) != 0:
        raise ValueError(f"trajectory lacks e3=0 baseline: {path}")
    return records


def replay(seed, path, threshold):
    records = load_trajectory(path)
    baseline_record = records[0]
    references = baseline_record.get("unimodal_references") or {}
    try:
        left_accuracy = float(references["left"]["validation_accuracy"])
        right_accuracy = float(references["right"]["validation_accuracy"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"trajectory lacks validated unimodal references: {path}"
        ) from error

    config = Phase3StopConfig(
        decision_rule="relative_unimodal_parity",
        emergency_stop_mode="numerical_only",
        min_epochs=1,
        max_epochs=int(records[-1]["phase_epoch"]),
        parity_patience=2,
        recovery_fraction_threshold=float(threshold),
    )
    baseline = ModalityEvaluationResult.from_state_dict(
        baseline_record["metrics"]
    )
    stopper = Phase3RelativeUnimodalStopper(
        config,
        baseline,
        unimodal_left_accuracy=left_accuracy,
        unimodal_right_accuracy=right_accuracy,
    )
    decision = stopper.initialize_baseline(
        baseline_record["checkpoint_path"]
    )
    for raw_record in records[1:]:
        if decision.should_stop:
            break
        decision = stopper.update(
            ModalityEvaluationResult.from_state_dict(raw_record["metrics"]),
            raw_record["checkpoint_path"],
        )

    selected = decision.selected
    if selected is None:
        selected, selection_status = stopper.selection()
    else:
        selection_status = decision.selection_status
    if selected is None:
        raise RuntimeError(f"stopper did not select a checkpoint: seed={seed}")
    if not Path(selected.checkpoint_path).is_file():
        raise FileNotFoundError(selected.checkpoint_path)
    return {
        "threshold": float(threshold),
        "seed": int(seed),
        "stop_epoch": int(decision.current.metrics.phase_epoch),
        "selected_epoch": int(selected.metrics.phase_epoch),
        "stop_reason": decision.stop_reason,
        "selection_status": selection_status,
        "selected_checkpoint": selected.checkpoint_path,
        "selected_recovery_fraction": selected.recovery_fraction,
        "selected_parity_gap": selected.parity_gap,
        "selected_weak_accuracy": selected.metrics.weak_only.accuracy,
        "selected_dominant_accuracy": (
            selected.metrics.dominant_only.accuracy
        ),
        "selected_full_accuracy": selected.metrics.full.accuracy,
        "baseline_weak_accuracy": baseline.weak_only.accuracy,
        "baseline_dominant_accuracy": baseline.dominant_only.accuracy,
        "baseline_full_accuracy": baseline.full.accuracy,
        "unimodal_left_accuracy": left_accuracy,
        "unimodal_right_accuracy": right_accuracy,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="SEED=/absolute/path/to/phase3_trajectory.jsonl",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trajectories = {}
    for value in args.trajectory:
        seed_text, path = value.split("=", 1)
        trajectories[int(seed_text)] = path
    rows = [
        replay(seed, trajectories[seed], threshold)
        for threshold in args.threshold
        for seed in sorted(trajectories)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "recovery_fraction_replay.csv", rows)
    (output_dir / "recovery_fraction_replay.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

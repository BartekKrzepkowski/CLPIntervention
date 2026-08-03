#!/usr/bin/env python3
"""Replay the local-accuracy stopper from scalar W&B Phase-3 history.

This is a point-estimate replay. Historical runs created before accuracy
stopping was introduced do not contain aligned per-example correctness at
every look or gradient-probe metrics. The script identifies a candidate
stopping interval without claiming an exact paired-CI reproduction.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb


REQUIRED_KEYS = (
    "phase3/phase_epoch",
    "phase3/weak_only_accuracy",
    "phase3/full_val_accuracy",
    "phase3/dominant_only_val_accuracy",
)
BASELINE_KEYS = (
    "phase3_baseline/weak_only_val_accuracy",
    "phase3_baseline/full_val_accuracy",
    "phase3_baseline/dominant_only_val_accuracy",
)


def _history(run, baseline_override=None):
    keys = [
        *REQUIRED_KEYS,
        *([] if baseline_override is not None else BASELINE_KEYS),
    ]
    baseline = {}
    by_step = {}
    for row in run.scan_history(keys=["_step", *keys], page_size=1000):
        step = row.get("_step")
        if step is not None:
            merged = by_step.setdefault(int(step), {})
            merged.update(
                {
                    key: value
                    for key, value in row.items()
                    if value is not None
                }
            )
    evaluations = {}
    for row in by_step.values():
        for key in BASELINE_KEYS:
            value = row.get(key)
            if value is not None:
                baseline[key] = float(value)
        if all(row.get(key) is not None for key in REQUIRED_KEYS):
            epoch = int(row["phase3/phase_epoch"])
            evaluations[epoch] = {
                "phase_epoch": epoch,
                "weak_accuracy": float(row["phase3/weak_only_accuracy"]),
                "full_accuracy": float(row["phase3/full_val_accuracy"]),
                "dominant_accuracy": float(
                    row["phase3/dominant_only_val_accuracy"]
                ),
            }
    if baseline_override is not None:
        baseline = {
            key: float(value)
            for key, value in zip(BASELINE_KEYS, baseline_override)
        }
    missing = [key for key in BASELINE_KEYS if key not in baseline]
    if missing:
        raise ValueError(f"{run.path}: missing baseline keys: {missing}")
    if not evaluations:
        raise ValueError(f"{run.path}: no complete Phase-3 evaluations")
    return baseline, [evaluations[key] for key in sorted(evaluations)]


def _slope(rows, key):
    epochs = [row["phase_epoch"] for row in rows]
    values = [row[key] for row in rows]
    epoch_mean = sum(epochs) / len(epochs)
    value_mean = sum(values) / len(values)
    denominator = sum((epoch - epoch_mean) ** 2 for epoch in epochs)
    if denominator == 0:
        return 0.0
    return sum(
        (epoch - epoch_mean) * (value - value_mean)
        for epoch, value in zip(epochs, values)
    ) / denominator


def _dominates(candidate, current, min_delta):
    keys = ("weak_accuracy", "full_accuracy", "dominant_accuracy")
    return (
        all(
            candidate[key] >= current[key] - min_delta
            for key in keys
        )
        and any(
            candidate[key] > current[key] + min_delta
            for key in keys
        )
    )


def replay(
    evaluations,
    baseline,
    *,
    minimum_exposure_evaluations,
    trend_window,
    target_patience,
    pareto_patience,
    futility_harm_patience,
    max_weak_accuracy_slope,
    min_weak_gain,
    min_delta,
):
    weak_baseline = baseline[BASELINE_KEYS[0]]
    full_baseline = baseline[BASELINE_KEYS[1]]
    dominant_baseline = baseline[BASELINE_KEYS[2]]
    frontier = []
    last_candidate = None
    futility_anchor = None
    target_checks = 0
    pareto_checks = 0
    futility_harm_checks = 0
    stop_epoch = None
    stop_reason = None
    selected_epoch = None
    rows = []
    for values in evaluations:
        row = dict(values)
        row["weak_accuracy_gain"] = row["weak_accuracy"] - weak_baseline
        row["full_accuracy_change"] = row["full_accuracy"] - full_baseline
        row["dominant_accuracy_change"] = (
            row["dominant_accuracy"] - dominant_baseline
        )
        history_window = [*rows, row][-trend_window:]
        enough_exposure = (
            len(rows) + 1 >= minimum_exposure_evaluations
            and len(history_window) >= trend_window
        )
        weak_slope = (
            _slope(history_window, "weak_accuracy")
            if enough_exposure
            else None
        )
        full_slope = (
            _slope(history_window, "full_accuracy")
            if enough_exposure
            else None
        )
        dominant_slope = (
            _slope(history_window, "dominant_accuracy")
            if enough_exposure
            else None
        )
        dominators = [
            candidate
            for candidate in frontier
            if _dominates(candidate, row, min_delta)
        ]
        dominator = (
            max(
                dominators,
                key=lambda item: (
                    item["weak_accuracy"],
                    item["full_accuracy"],
                    item["dominant_accuracy"],
                    -item["phase_epoch"],
                ),
            )
            if dominators
            else None
        )
        target = bool(
            enough_exposure
            and dominator is None
            and row["weak_accuracy_gain"] >= min_weak_gain
            and row["weak_accuracy"] >= row["dominant_accuracy"]
            and full_slope >= 0.0
            and dominant_slope >= 0.0
        )
        pareto_reversal = bool(enough_exposure and dominator is not None)
        weak_futile = bool(
            enough_exposure
            and weak_slope <= max_weak_accuracy_slope
        )
        accuracy_harm = bool(
            enough_exposure
            and (full_slope < 0.0 or dominant_slope < 0.0)
        )
        futility_with_harm = weak_futile and accuracy_harm
        target_checks = target_checks + 1 if target else 0
        pareto_checks = pareto_checks + 1 if pareto_reversal else 0
        if futility_with_harm and futility_harm_checks == 0:
            futility_anchor = last_candidate
        futility_harm_checks = (
            futility_harm_checks + 1 if futility_with_harm else 0
        )
        if dominator is None and not accuracy_harm:
            last_candidate = row
        if not any(
            _dominates(candidate, row, min_delta)
            for candidate in frontier
        ):
            frontier = [
                candidate
                for candidate in frontier
                if not _dominates(row, candidate, min_delta)
            ]
            frontier.append(row)
            frontier = sorted(
                frontier, key=lambda item: item["phase_epoch"]
            )[-(trend_window + pareto_patience) :]
        row.update(
            {
                "weak_accuracy_slope": weak_slope,
                "full_accuracy_slope": full_slope,
                "dominant_accuracy_slope": dominant_slope,
                "target": target,
                "pareto_reversal": pareto_reversal,
                "weak_futile": weak_futile,
                "accuracy_harm": accuracy_harm,
                "futility_with_harm": futility_with_harm,
                "target_checks": target_checks,
                "pareto_checks": pareto_checks,
                "futility_harm_checks": futility_harm_checks,
                "dominator_epoch": (
                    None if dominator is None else dominator["phase_epoch"]
                ),
            }
        )
        if stop_epoch is None and target_checks >= target_patience:
            stop_epoch = row["phase_epoch"]
            stop_reason = "target_reached"
            selected_epoch = row["phase_epoch"]
        elif stop_epoch is None and pareto_checks >= pareto_patience:
            stop_epoch = row["phase_epoch"]
            stop_reason = "pareto_reversal"
            selected_epoch = dominator["phase_epoch"]
        elif (
            stop_epoch is None
            and futility_harm_checks >= futility_harm_patience
        ):
            stop_epoch = row["phase_epoch"]
            stop_reason = "futility_with_harm"
            selected_epoch = (
                None
                if futility_anchor is None
                else futility_anchor["phase_epoch"]
            )
        rows.append(row)
    return {
        "stop_reason": stop_reason or "max_epochs",
        "stop_epoch_point_estimate": stop_epoch,
        "selected_epoch_point_estimate": selected_epoch,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="SEED=entity/project/run_id; repeat for each paired seed",
    )
    parser.add_argument(
        "--input-json",
        help="Reuse a previous replay JSON instead of scanning W&B",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="SEED=weak,full,dominant for resumed runs without baseline logs",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--minimum-exposure-evaluations", type=int, default=5)
    parser.add_argument("--trend-window", type=int, default=5)
    parser.add_argument("--target-patience", type=int, default=2)
    parser.add_argument("--pareto-patience", type=int, default=2)
    parser.add_argument("--futility-harm-patience", type=int, default=2)
    parser.add_argument("--max-weak-accuracy-slope", type=float, default=0.0)
    parser.add_argument("--min-weak-gain", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=0.001)
    args = parser.parse_args()

    if not args.run and not args.input_json:
        parser.error("provide --run or --input-json")
    api = wandb.Api() if args.run else None
    baseline_overrides = {}
    for value in args.baseline:
        seed, raw = value.split("=", 1)
        parsed = tuple(float(item) for item in raw.split(","))
        if len(parsed) != 3:
            raise ValueError("baseline requires weak,full,dominant")
        baseline_overrides[seed] = parsed
    output = {"method": "local_accuracy_point_estimate_replay", "runs": {}}
    csv_rows = []
    sources = []
    if args.input_json:
        cached = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        for seed, run in cached["runs"].items():
            sources.append(
                (
                    seed,
                    run.get("wandb_run"),
                    run["baseline"],
                    [
                        {
                            key: row[key]
                            for key in (
                                "phase_epoch",
                                "weak_accuracy",
                                "full_accuracy",
                                "dominant_accuracy",
                            )
                        }
                        for row in run["rows"]
                    ],
                )
            )
    for value in args.run:
        seed, run_path = value.split("=", 1)
        baseline, evaluations = _history(
            api.run(run_path), baseline_overrides.get(seed)
        )
        sources.append((seed, run_path, baseline, evaluations))
    for seed, run_path, baseline, evaluations in sources:
        result = replay(
            evaluations,
            baseline,
            minimum_exposure_evaluations=(
                args.minimum_exposure_evaluations
            ),
            trend_window=args.trend_window,
            target_patience=args.target_patience,
            pareto_patience=args.pareto_patience,
            futility_harm_patience=args.futility_harm_patience,
            max_weak_accuracy_slope=args.max_weak_accuracy_slope,
            min_weak_gain=args.min_weak_gain,
            min_delta=args.min_delta,
        )
        output["runs"][seed] = {
            "wandb_run": run_path,
            "baseline": baseline,
            **result,
        }
        for row in result["rows"]:
            csv_rows.append({"seed": int(seed), **row})

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

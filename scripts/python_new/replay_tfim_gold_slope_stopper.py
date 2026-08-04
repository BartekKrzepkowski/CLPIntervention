#!/usr/bin/env python3
"""Replay a gold-slope TFIM stopper without recovery-oracle inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.python_new.analyze_tfim_p4_refinement import (
    FULL_EPOCHS,
    _find_log_in_directories,
    _phase_summary,
    _tfim_trace,
)
from scripts.python_new.tfim_analysis_common import probe_features


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", required=True, metavar="JOB:SEED:E3")
    parser.add_argument("--gold", action="append", required=True, metavar="JOB:SEED")
    parser.add_argument("--slurm-log-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _split(value, fields):
    parts = value.split(":")
    if len(parts) != fields:
        raise ValueError(f"invalid run specification: {value!r}")
    return parts


def _load_slope(log_dirs, job_id, seed, e3, *, phase, role):
    log_path = _find_log_in_directories(log_dirs, job_id)
    summary = _phase_summary(log_path, phase)
    if int(summary["executed_epochs"]) < FULL_EPOCHS[-1]:
        raise ValueError(f"job {job_id} ended before e{FULL_EPOCHS[-1]}")
    trace_path, trace = _tfim_trace(summary, phase=phase, epochs=FULL_EPOCHS)
    ratios = [trace[epoch]["ratio"] for epoch in FULL_EPOCHS]
    slope = probe_features(FULL_EPOCHS, ratios)["slope_log_ratio"]
    return {
        "job_id": int(job_id),
        "seed": int(seed),
        "e3": int(e3),
        "role": role,
        "slope_log_ratio": slope,
        **{f"ratio_e{epoch}": trace[epoch]["ratio"] for epoch in FULL_EPOCHS},
        "slurm_log": log_path.name,
        "tfim_trace_artifact": trace_path.name,
    }


def _sign(value):
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def replay_seed(rows, gold_slope):
    ordered = sorted(rows, key=lambda row: row["e3"])
    if not ordered:
        raise ValueError("a seed requires at least one P3 probe")
    if len({row["e3"] for row in ordered}) != len(ordered):
        raise ValueError("duplicate P3 probe epoch")

    revealed = []
    initial_side = None
    previous = None
    result = {
        "status": "no_crossing_observed",
        "initial_side": None,
        "bracket_left_e3": None,
        "bracket_right_e3": None,
        "selected_e3": None,
    }
    for row in ordered:
        current = dict(row)
        current["gold_slope_log_ratio"] = gold_slope
        current["delta_slope_to_gold"] = current["slope_log_ratio"] - gold_slope
        current["absolute_delta_slope_to_gold"] = abs(current["delta_slope_to_gold"])
        side = _sign(current["delta_slope_to_gold"])
        if initial_side is None and side != 0:
            initial_side = side
            result["initial_side"] = "above_gold" if side > 0 else "below_gold"

        if side == 0:
            current["decision"] = "stop_exact_gold_slope"
            result.update(status="exact_match", selected_e3=current["e3"])
            revealed.append(current)
            break
        if previous is not None and _sign(previous["delta_slope_to_gold"]) != side:
            current["decision"] = "stop_and_refine_first_crossing"
            result.update(
                status="crossing_bracket_found",
                bracket_left_e3=previous["e3"],
                bracket_right_e3=current["e3"],
            )
            revealed.append(current)
            break

        current["decision"] = "continue_p3"
        revealed.append(current)
        previous = current

    if result["status"] == "no_crossing_observed":
        result["closest_observed_e3_diagnostic_only"] = min(
            revealed,
            key=lambda row: (row["absolute_delta_slope_to_gold"], row["e3"]),
        )["e3"]
    result["revealed"] = revealed
    return result


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _arguments()
    probes = [
        _load_slope(
            args.slurm_log_dir,
            *_split(specification, 3),
            phase=4,
            role="counterfactual_p4_17_probe",
        )
        for specification in args.probe
    ]
    gold_rows = [
        _load_slope(
            args.slurm_log_dir,
            *_split(specification, 2),
            e3=0,
            phase=2,
            role="clean_p2_gold",
        )
        for specification in args.gold
    ]
    gold = {row["seed"]: row for row in gold_rows}
    seeds = {row["seed"] for row in probes}
    if set(gold) != seeds:
        raise ValueError("every probe seed requires exactly one matching gold run")

    replays = {
        str(seed): replay_seed(
            [row for row in probes if row["seed"] == seed],
            gold[seed]["slope_log_ratio"],
        )
        for seed in sorted(seeds)
    }
    revealed_rows = [
        row
        for seed in sorted(replays, key=int)
        for row in replays[seed]["revealed"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "tfim_gold_slope_stopper_replay.csv", revealed_rows)
    _write_csv(args.output_dir / "tfim_gold_slope_gold.csv", gold_rows)
    with (args.output_dir / "tfim_gold_slope_stopper_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "probe_epochs": FULL_EPOCHS,
                "target": "per-seed clean P2 gold slope",
                "decision_rule": "continue until the first sign crossing of probe slope minus gold slope",
                "oracle_recovery_metrics_used": False,
                "test_metrics_used": False,
                "replays": replays,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()

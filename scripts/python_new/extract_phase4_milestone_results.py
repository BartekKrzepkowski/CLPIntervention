#!/usr/bin/env python3
"""Extract validation-selected Phase-4 summaries from local Slurm logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN_PATTERN = re.compile(r"/runs/([a-zA-Z0-9]+)")


def extract(job_spec, log_dir):
    seed, e3, job_id = job_spec.split(":", 2)
    path = log_dir / f"experiment-{job_id}.out"
    summary = None
    run_id = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Phase summary: " in line:
            summary = json.loads(line.split("Phase summary: ", 1)[1])
        match = RUN_PATTERN.search(line)
        if match:
            run_id = match.group(1)
    if summary is None:
        raise ValueError(f"{path}: no completed Phase-4 summary")
    validation = summary["best_full_accuracy_metrics"]
    budget_validation = summary["best_budget_matched_accuracy_metrics"]
    test = summary["test_metrics"]["best_full_accuracy"]
    budget_test = summary["test_metrics"]["best_budget_matched_accuracy"]
    return {
        "seed": int(seed),
        "e3": int(e3),
        "slurm_job_id": int(job_id),
        "wandb_run_id": run_id,
        "selected_p4_epoch": validation["phase_epoch"],
        "val_full_accuracy": validation["full_accuracy"],
        "val_full_loss": validation["full_loss"],
        "val_dominant_accuracy": validation["dominant_only_accuracy"],
        "val_weak_accuracy": validation["weak_only_accuracy"],
        "test_proper_accuracy": test["proper_accuracy"],
        "test_proper_loss": test["proper_loss"],
        "test_blurred_accuracy": test["blurred_accuracy"],
        "test_blurred_loss": test["blurred_loss"],
        "budget_selected_p4_epoch": budget_validation["phase_epoch"],
        "budget_val_full_accuracy": budget_validation["full_accuracy"],
        "budget_test_proper_accuracy": budget_test["proper_accuracy"],
        "budget_test_blurred_accuracy": budget_test["blurred_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        action="append",
        required=True,
        help="SEED:E3:SLURM_JOB_ID; repeat for every milestone",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("slurm_logs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = sorted(
        (extract(spec, args.log_dir) for spec in args.job),
        key=lambda row: (row["seed"], row["e3"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

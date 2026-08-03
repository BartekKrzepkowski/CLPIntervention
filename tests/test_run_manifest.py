import json
import sys
from pathlib import Path

import pytest

from scripts.python_new import run_single
from src.utils.run_manifest import RunManifest, sha256_dataset, sha256_file


def test_run_manifest_hashes_dataset_source_checkpoint_and_artifacts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("source", encoding="utf-8")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "part-b.bin").write_bytes(b"b")
    (dataset / "part-a.bin").write_bytes(b"a")
    checkpoint = tmp_path / "input.pth"
    checkpoint.write_bytes(b"input")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    manifest = RunManifest.create(
        run_dir,
        config={"seed": 83},
        repo_root=repo,
        dataset_path=dataset,
        input_checkpoint=checkpoint,
    )
    artifact = run_dir / "checkpoint.pth"
    artifact.write_bytes(b"output")
    manifest.add_artifact(artifact, "checkpoint")
    manifest.finalize("completed")

    payload = json.loads((run_dir / "run_manifest.json").read_text())
    assert payload["format"] == "clpintervention.run"
    assert payload["status"] == "completed"
    assert payload["dataset"] == sha256_dataset(dataset)
    assert payload["input_checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert payload["source"]["tree_files"] == 1
    assert payload["source"]["tree_sha256"]
    assert payload["environment"]["packages"]
    assert payload["artifacts"] == [
        {
            "kind": "checkpoint",
            "path": "checkpoint.pth",
            "bytes": 6,
            "sha256": sha256_file(artifact),
        }
    ]
    assert not (run_dir / "run_manifest.json.tmp").exists()


def test_dataset_hash_depends_on_relative_names_and_content(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    path = dataset / "sample.bin"
    path.write_bytes(b"first")
    first = sha256_dataset(dataset)["sha256"]
    path.write_bytes(b"second")
    second = sha256_dataset(dataset)["sha256"]
    assert first != second


def test_repository_publication_config_is_frozen():
    config = run_single.OmegaConf.load(
        "configs/experiments/paper_cifar10_sresnet18_baseline_seed83.yaml"
    )
    assert config.frozen is True
    assert config.protocol_role == "baseline"
    assert config.model_name == "mm_resnet"
    assert config.dataset_name == "mm_cifar10"
    assert config.seed == 83
    assert [config.phase1, config.phase2, config.phase3, config.phase4] == [
        80, 200, 96, 200
    ]
    assert config.fim_probe_fraction == 0.02
    assert config.fim_measurements_per_epoch == 2
    assert config.fim_samples_per_input == 5
    assert config.phase4_bn_recalibration_batches == 0
    assert config.phase4_weight_decay is None
    assert config.phase4_lr_lambda is None


def test_frozen_config_loads_without_scientific_cli_overrides(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(
        "frozen: true\nfrozen_config_version: 1\nmode: all_at_once\nseed: 83\n",
        encoding="utf-8",
    )
    captured = {}
    monkeypatch.setattr(
        run_single,
        "run",
        lambda mode, config, umt=False: captured.update(
            mode=mode, config=config, umt=umt
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_single", f"frozen_config={frozen}"])

    run_single.main()

    assert captured["mode"] == "all_at_once"
    assert captured["config"].seed == 83
    assert captured["config"].frozen_config_path == str(frozen.resolve())


def test_frozen_config_rejects_scientific_cli_overrides(monkeypatch, tmp_path):
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(
        "frozen: true\nfrozen_config_version: 1\nmode: all_at_once\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_single", f"frozen_config={frozen}", "lr=0.1"],
    )

    with pytest.raises(ValueError, match="reject scientific CLI overrides"):
        run_single.main()

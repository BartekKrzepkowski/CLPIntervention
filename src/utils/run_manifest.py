import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


MANIFEST_FORMAT = "clpintervention.run"
MANIFEST_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataset(path):
    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    digest = hashlib.sha256()
    files = (
        [root]
        if root.is_file()
        else sorted(item for item in root.rglob("*") if item.is_file())
    )
    total_bytes = 0
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        size = item.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(root),
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def _git_value(repo_root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_tree_identity(repo_root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    paths = sorted(item.decode("utf-8") for item in result.stdout.split(b"\0") if item)
    digest = hashlib.sha256()
    included = 0
    for relative in paths:
        path = Path(repo_root, relative)
        if not path.is_file():
            continue
        included += 1
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"sha256": digest.hexdigest(), "files": included}


def source_identity(repo_root):
    repo_root = str(Path(repo_root).resolve())
    status = (
        _git_value(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
        or ""
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    ).stdout
    source_tree = _source_tree_identity(repo_root)
    return {
        "repo_root": repo_root,
        "commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "branch": _git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tree_sha256": source_tree["sha256"],
        "tree_files": source_tree["files"],
    }


def environment_identity():
    return {
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in sorted(
                metadata.distributions(),
                key=lambda item: (item.metadata["Name"] or "").lower(),
            )
            if distribution.metadata["Name"]
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
    }


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class RunManifest:
    def __init__(self, run_dir, payload):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "run_manifest.json"
        self.payload = payload
        self.write()

    @classmethod
    def create(
        cls, run_dir, config, repo_root, dataset_path, input_checkpoint=None
    ):
        checkpoint = None
        if input_checkpoint:
            resolved = Path(input_checkpoint).resolve()
            checkpoint = {"path": str(resolved), "sha256": sha256_file(resolved)}
        payload = {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            "status": "running",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            "source": source_identity(repo_root),
            "environment": environment_identity(),
            "dataset": sha256_dataset(dataset_path),
            "input_checkpoint": checkpoint,
            "config": config,
            "artifacts": [],
        }
        return cls(run_dir, payload)

    def write(self):
        self.payload["updated_at"] = _utc_now()
        _atomic_json(self.path, self.payload)

    def add_artifact(self, path, kind):
        artifact = Path(path).resolve()
        entry = {
            "kind": kind,
            "path": os.path.relpath(artifact, self.run_dir),
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
        self.payload["artifacts"] = [
            item
            for item in self.payload["artifacts"]
            if item["path"] != entry["path"]
        ]
        self.payload["artifacts"].append(entry)
        self.write()

    def finalize(self, status, error=None):
        self.payload["status"] = status
        self.payload["finished_at"] = _utc_now()
        if error is not None:
            self.payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self.write()

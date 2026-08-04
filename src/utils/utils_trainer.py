import os
import random
from datetime import datetime

import numpy as np
import torch


def manual_seed(random_seed, device):
    """Seed Python, NumPy, and PyTorch for a new run."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(random_seed)


def adjust_evaluators(d1, dd2, denom, scope, phase):
    for evaluator_key in dd2:
        eval_key_split_1 = str(evaluator_key).split('/')
        if len(eval_key_split_1) == 1:
            d1[f'{scope}_{eval_key_split_1[0]}/{phase}'] += dd2[evaluator_key] * denom
        else:
            if '____' not in eval_key_split_1[1]:
                eval_key_split_1[1] += f'____{phase}'

            eval_key_split_2 = eval_key_split_1[0].split('_')
            if eval_key_split_2[0] in {'running', 'epoch'}:
                eval_key_split_2 = [scope] + eval_key_split_2[1:]
            else:
                eval_key_split_2 = [scope] + eval_key_split_2
            eval_key_split_1[0] = '_'.join(eval_key_split_2)
            d1['/'.join(eval_key_split_1)] += dd2[evaluator_key] * denom
        
        # eval_key = str(evaluator_key).split('/')
        # if 'train' in evaluator_key or 'valid' in evaluator_key or 'test' in evaluator_key:
        #     eval_key = '/'.join(eval_key[:-1]).split('_')
        #     eval_key = '_'.join(eval_key[1:]) if eval_key[0] in {'running', 'epoch'} else '_'.join(eval_key)
        #     d1[f'{scope}_{eval_key}/{phase}'] += dd2[evaluator_key] * denom
        # elif len(eval_key) == 1:
        #     d1[f'{scope}_{eval_key[0]}/{phase}'] += dd2[evaluator_key] * denom
        # else:
        #     eval_key = '/'.join(eval_key).split('_')
        #     eval_key = '_'.join(eval_key[1:]) if eval_key[0] in {'running', 'epoch'} else '_'.join(eval_key)
        #     d1[f'{scope}_{eval_key}'] += dd2[evaluator_key] * denom
    return d1


def adjust_evaluators_pre_log(d1, denom, round_at=None):
    """Average metrics without altering raw values unless display rounding is requested."""
    if denom == 0:
        raise ValueError("Cannot average metrics with a zero denominator")
    averaged = {key: value / denom for key, value in d1.items()}
    if round_at is None:
        return averaged
    return {key: round(value, round_at) for key, value in averaged.items()}


def update_tensor(a, b):
    c = torch.cat([a, b])
    return c


def find_paths(path):
    dirs = os.listdir(path)
    path = os.path.join(path, dirs[-1], 'checkpoints')
    dirs = sorted([os.path.join(path, dir) for dir in os.listdir(path)])
    return dirs


CHECKPOINT_FORMAT = "clpintervention.training"
CHECKPOINT_VERSION = 5


def _load_state_dict(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def load_checkpoint_metadata(path, device="cpu"):
    """Read versioned checkpoint metadata without constructing a model."""
    payload = _load_state_dict(path, device)
    if not (
        isinstance(payload, dict)
        and payload.get("format") == CHECKPOINT_FORMAT
    ):
        raise ValueError(
            "unimodal references require a versioned training checkpoint"
        )
    return {
        "version": int(payload.get("version", 0)),
        "metadata": dict(payload.get("metadata") or {}),
        "phase_epoch": payload.get("phase_epoch"),
        "next_epoch": payload.get("next_epoch"),
        "global_step": payload.get("global_step"),
    }


def _model_state(payload):
    if (
        isinstance(payload, dict)
        and payload.get("format") == CHECKPOINT_FORMAT
        and "model_state_dict" in payload
    ):
        return payload["model_state_dict"]
    return payload


def _load_model_state(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as direct_error:
        if hasattr(model, "main_model"):
            try:
                model.main_model.load_state_dict(state_dict)
                return
            except RuntimeError:
                pass
        prefix = "main_model."
        main_state = {
            name[len(prefix):]: value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        if main_state:
            target = model.main_model if hasattr(model, "main_model") else model
            target.load_state_dict(main_state)
            return
        raise direct_error


def load_model(model, path, device=None):
    if device is None:
        device = next(model.parameters()).device
    _load_model_state(model, _model_state(_load_state_dict(path, device)))
    return model


def load_branch(model, path, branch_name, device=None):
    """Load a branch-only, bimodal, UMT, or versioned training checkpoint."""
    if device is None:
        device = next(model.parameters()).device
    state_dict = _model_state(_load_state_dict(path, device))
    prefixes = (f"{branch_name}.", f"main_model.{branch_name}.")
    branch_state = {}
    for prefix in prefixes:
        branch_state = {
            name[len(prefix):]: value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        if branch_state:
            break
    model.load_state_dict(branch_state or state_dict)
    return model



def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    *,
    next_epoch,
    global_step,
    metadata=None,
    diagnostics_state=None,
    phase_state=None,
    loader_state=None,
    phase_epoch=None,
):
    scheduler_multipliers = None
    if scheduler is not None and hasattr(scheduler, "lr_lambdas"):
        try:
            scheduler_multipliers = [
                float(lr_lambda(0)) for lr_lambda in scheduler.lr_lambdas
            ]
        except (TypeError, ValueError):
            scheduler_multipliers = None
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scheduler_multipliers": scheduler_multipliers,
        "next_epoch": int(next_epoch),
        "global_step": int(global_step),
        "rng_state": capture_rng_state(),
        "metadata": dict(metadata or {}),
        "diagnostics_state": diagnostics_state,
        "phase_state": phase_state,
        "loader_state": loader_state,
        "phase_epoch": None if phase_epoch is None else int(phase_epoch),
    }
    path = os.fspath(path)
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_training_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    payload = _load_state_dict(path, device)
    _load_model_state(model, _model_state(payload))

    is_training_checkpoint = (
        isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT
    )
    if not is_training_checkpoint:
        return {
            "is_training_checkpoint": False,
            "next_epoch": None,
            "global_step": None,
            "rng_state": None,
            "metadata": {},
            "diagnostics_state": None,
            "phase_state": None,
            "loader_state": None,
            "phase_epoch": None,
        }
    version = int(payload.get("version", 0))
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {version} is newer than supported "
            f"version {CHECKPOINT_VERSION}."
        )
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        multipliers = payload.get("scheduler_multipliers")
        if multipliers is not None and hasattr(scheduler, "lr_lambdas"):
            scheduler.lr_lambdas = [
                lambda _epoch, factor=factor: factor for factor in multipliers
            ]
    return {
        "is_training_checkpoint": True,
        "next_epoch": int(payload["next_epoch"]),
        "global_step": int(payload["global_step"]),
        "rng_state": payload.get("rng_state"),
        "metadata": dict(payload.get("metadata", {})),
        "diagnostics_state": payload.get("diagnostics_state"),
        "phase_state": payload.get("phase_state"),
        "loader_state": payload.get("loader_state"),
        "phase_epoch": payload.get("phase_epoch"),
    }


def create_paths(base_path, exp_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    reports_root = os.path.abspath(base_path or ".")
    run_path = os.path.join(reports_root, exp_name, timestamp)
    checkpoint_path = os.path.join(run_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=False)

    def save_path(step):
        return os.path.join(checkpoint_path, f"model_step_{step}.pth")

    return run_path, save_path


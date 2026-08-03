#!/usr/bin/env python3
"""Compute post-hoc RSV tensors for one or more phase checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.analysis.rsv import (
    DEFAULT_RSV_LAYER_SPECS,
    RSVLayerSpec,
    RSVProbeConfig,
    measure_rsv_layers,
    save_rsv_result,
)
from src.data.probes import _raw_dataset
from src.utils.prepare import prepare_loaders_clp, prepare_model
from src.utils.utils_data import count_classes
from src.utils.utils_model import load_model_specific_params
from src.utils.utils_trainer import load_training_checkpoint


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_specs(values):
    specs = []
    seen = set()
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).stem
        if not label or not path:
            raise ValueError("--checkpoint must use [LABEL=]PATH")
        if label in seen:
            raise ValueError(f"duplicate checkpoint label: {label}")
        seen.add(label)
        specs.append((label, Path(path)))
    return specs


def _output_path(args, checkpoint_label, measurement, checkpoint_count):
    if args.output_dir is not None:
        return Path(args.output_dir) / f"{checkpoint_label}.{measurement}.pt"
    if args.output is None:
        raise ValueError("provide --output-dir or --output")
    if checkpoint_count != 1:
        raise ValueError("--output supports one checkpoint; use --output-dir")
    output = Path(args.output)
    if args.layer is not None:
        return output
    if output.suffix:
        return output.with_name(f"{output.stem}.{measurement}{output.suffix}")
    return Path(f"{output}.{measurement}.pt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="[PHASE=]PATH",
        help="repeat for phase1..phase4 checkpoints",
    )
    parser.add_argument("--output")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-name", default="mm_resnet")
    parser.add_argument("--dataset-name", default="mm_cifar10")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--resize-factor", type=float, default=0.25)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--variants-per-source", type=int, default=100)
    parser.add_argument("--translate-pixels", type=int, default=4)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--layer",
        help="custom single layer; default records stage3_avgpool and stage4_avgpool",
    )
    parser.add_argument(
        "--spatial-average-pool",
        action="store_true",
        help="apply analysis-only adaptive average pooling to a custom layer",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_specs = _checkpoint_specs(args.checkpoint)
    if args.output is not None and args.output_dir is not None:
        raise ValueError("--output and --output-dir are mutually exclusive")
    if args.layer is None and args.spatial_average_pool:
        raise ValueError("--spatial-average-pool requires --layer")
    layer_specs = (
        (
            RSVLayerSpec(
                "custom",
                args.layer,
                spatial_average_pool=args.spatial_average_pool,
            ),
        )
        if args.layer is not None
        else DEFAULT_RSV_LAYER_SPECS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = prepare_loaders_clp(
        args.dataset_name,
        dataset_params={
            "dataset_path": None,
            "overlap": args.overlap,
            "resize_factor": args.resize_factor,
            "subset": None,
        },
        loader_params={"batch_size": 1, "num_workers": 0},
    )
    train_dataset = loaders["train"].dataset
    eval_dataset = loaders["test_proper"].dataset
    input_channels, img_height, img_width = eval_dataset[0][0][0].shape
    model_params = {
        "num_classes": count_classes(train_dataset),
        "input_channels": input_channels,
        "img_height": img_height,
        "img_width": img_width,
        "overlap": args.overlap,
        **load_model_specific_params(args.model_name),
    }
    model = prepare_model(args.model_name, model_params).to(device)
    probe_config = RSVProbeConfig(
        samples_per_class=args.samples_per_class,
        variants_per_source=args.variants_per_source,
        translate_pixels=args.translate_pixels,
        rotation_degrees=args.rotation_degrees,
        seed=args.seed,
        overlap=args.overlap,
    )
    outputs = []
    for checkpoint_label, checkpoint in checkpoint_specs:
        checkpoint = checkpoint.resolve()
        checkpoint_state = load_training_checkpoint(
            checkpoint, model, device=device
        )
        results = measure_rsv_layers(
            model,
            _raw_dataset(train_dataset),
            eval_dataset.transform1,
            eval_dataset.transform2,
            config=probe_config,
            layer_specs=layer_specs,
            batch_size=args.batch_size,
            device=device,
            extra_metadata={
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "checkpoint_label": checkpoint_label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_metadata": checkpoint_state["metadata"],
                "device": str(device),
            },
        )
        for measurement, result in results.items():
            output_path = _output_path(
                args, checkpoint_label, measurement, len(checkpoint_specs)
            )
            manifest_path = save_rsv_result(result, output_path)
            outputs.append(
                {
                    "checkpoint": checkpoint_label,
                    "measurement": measurement,
                    "raw": str(output_path.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "shape": list(result["rsv"].shape),
                }
            )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

import json


import numpy as np
import torch

from src.utils.utils_data import get_targets


def get_samples_weights(loaders, num_classes):
    targets = np.asarray(get_targets(loaders["train"].dataset), dtype=np.int64)
    class_counts = np.bincount(targets, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = class_counts > 0
    weights[present] = 1.0 / class_counts[present]
    return torch.from_numpy(weights)


def load_criterion_specific_params(criterion_name):
    with open(f"src/configs/{criterion_name}.json", encoding="utf-8") as config_file:
        return json.load(config_file)
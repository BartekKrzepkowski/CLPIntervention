import json
from collections import Counter

import torch


def get_samples_weights(loaders, num_classes):
    targets = loaders['train'].dataset.dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    _, class_counts = zip(*sorted(Counter(targets).items()))
    samples_weights = torch.tensor([1 / class_count for class_count in class_counts])
    return samples_weights


def load_criterion_specific_params(criterion_name):
    criterion_params = json.load(open(f'src/configs/{criterion_name}.json', 'r'))
    return criterion_params
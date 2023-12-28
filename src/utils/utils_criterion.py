from collections import Counter

import torch


def get_samples_weights(loaders, num_classes):
    targets = loaders['train'].dataset.dataset.targets
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    _, class_counts = zip(*sorted(Counter(targets).items()))
    samples_weights = torch.tensor([1 / class_count for class_count in class_counts])
    return samples_weights
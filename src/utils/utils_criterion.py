import torch


def get_samples_weights(loaders, num_classes):
    class_counts = [0] * num_classes
    for _, label in loaders['train'].dataset:
        class_counts[label] += 1
    samples_weights = torch.tensor([1 / class_counts[i] for i in range(len(class_counts))])
    return samples_weights
import os

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def count_classes(dataset):
    classes = len(np.unique(np.array(dataset.dataset.targets)))
    return classes

#write a function to calculate mean and std of cifar10
def get_mean_std(dataloader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_mean_std_all(dataloader):
    mean = 0.
    var = 0.
    pixel_count = 0.0
    nb_samples = 0.
    for x_data, _ in dataloader:
        batch_samples = x_data.size(0)
        x_data = x_data.view(batch_samples, x_data.size(1), -1)
        mean += x_data.mean(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    for x_data, _ in dataloader:
        batch_samples = x_data.size(0)
        x_data = x_data.view(batch_samples, x_data.size(1), -1)
        var += ((x_data - mean.unsqueeze(1))**2).sum([0,2])
        pixel_count += x_data.nelement()
    std = torch.sqrt(var / (pixel_count-1))
    return mean, std

import numpy as np

def get_mean_std_3(dataloader):
    mean = 0.0
    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)

    std = 0.0
    for images, _ in dataloader:
        images = images.view(images.size(0), images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(std / (len(dataloader.dataset)*32*32))
    return mean, std

def get_mean_std_4(dataset):
    x = np.stack([np.asarray(dataset[i][0]) for i in range(len(dataset))])
    train_mean = np.mean(x, axis=(0, 2, 3))
    train_std = np.std(x, axis=(0, 2, 3))
    return train_mean, train_std



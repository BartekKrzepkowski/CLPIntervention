import torch
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

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


from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import InterpolationMode
transform_basic = Compose([ToTensor()])
# # tensor([0.4909, 0.4816, 0.4459]) tensor([0.2162, 0.2135, 0.2342])
# transform_blurred = Compose([ToTensor(), Resize(8, interpolation=InterpolationMode.BILINEAR, antialias=None), Resize(32, interpolation=InterpolationMode.BILINEAR, antialias=None)])
dataset = datasets.CIFAR10(root=os.environ['CIFAR10_PATH'], train=True, download=True, transform=transform_basic)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
# mean, std = get_mean_std_3(dataloader)
# print(mean, std)



x = np.stack([np.asarray(dataset[i][0]) for i in range(len(dataset))])
# # print(x)
# print(x.shape)`
# # calculate the mean and std along the (0, 1) axes`
train_mean = np.mean(x, axis=(0, 2, 3))
train_std = np.std(x, axis=(0, 2, 3))
# # the the mean and std
print(train_mean, train_std)
print(train_mean.shape, train_std.shape)
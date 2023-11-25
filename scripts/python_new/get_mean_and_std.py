import math
import os

import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class NormalizationDataset(Dataset):
    def __init__(self, dataset, transform1, transform2, overlap=0.5):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.with_overlap = overlap / 2 + 0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Split the image into two halves
        # print(type(image))
        width, height = image.size
        width_ = math.ceil(width * self.with_overlap)
        image1 = image.crop((0, 0, width_, height))
        image2 = image.crop((width-width_, 0, width, height))

        image1 = self.transform1(image1)
        image2 = self.transform2(image2)

        return (image1, image2), label
    
    
if __name__ == '__main__':
    

    transform_train_proper = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_train_blurred = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((16, 8), interpolation=InterpolationMode.BILINEAR, antialias=None),
        transforms.Resize((64, 32), interpolation=InterpolationMode.BILINEAR, antialias=None),
    ])

    dataset_path = os.environ['TINYIMAGENET_PATH']
    train_path = f'{dataset_path}/train'

    dataset = datasets.ImageFolder(train_path)
    dataset1 = NormalizationDataset(dataset, transform1=transform_train_proper, transform2=transform_train_blurred, overlap=0.0)
    dataset2 = NormalizationDataset(dataset, transform1=transform_train_proper, transform2=transform_train_proper, overlap=0.0)
    
    x = torch.stack([item[0][1] for item in dataset1], dim=0).numpy()

    train_mean = np.mean(x, axis=(0, 2, 3))
    train_std = np.std(x, axis=(0, 2, 3))

    print("Right branch blurred", train_mean, train_std)
    
    x = torch.stack([item[0][1] for item in dataset2], dim=0).numpy()

    train_mean = np.mean(x, axis=(0, 2, 3))
    train_std = np.std(x, axis=(0, 2, 3))

    print("Right branch proper", train_mean, train_std)
    
    x = torch.stack([item[0][0] for item in dataset2], dim=0).numpy()
    train_mean = np.mean(x, axis=(0, 2, 3))
    train_std = np.std(x, axis=(0, 2, 3))

    print("Left branch proper", train_mean, train_std)
    
    

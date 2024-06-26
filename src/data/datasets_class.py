import logging

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image

class DualAugmentDataset(Dataset):
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image1 = self.transform1(image)
        image2 = self.transform2(image)

        return image1, image2, label
    
import math
class SplitAndAugmentDataset(Dataset):
    def __init__(self, dataset, transform1, transform2, transform3=None, subset=tuple(), overlap=0.5, is_train=True, reverse=True):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        self.subset = subset
        self.with_overlap = overlap / 2 + 0.5
        self.is_train = is_train
        self.reverse = reverse
        logging.info(f'Overlap between visual fields: {overlap * 100} %, with a single visual field: {self.with_overlap * 100} % of the whole.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Split the image into two halves with overlap
        if self.is_train and self.reverse:
            if torch.rand(1) > 0.5:
                # reverse image horizontally
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = image.size
        width_ = math.ceil(width * self.with_overlap)
        image1 = image.crop((0, 0, width_, height))
        image2 = image.crop((width-width_, 0, width, height))

        image1 = self.transform1(image1)
        if self.is_train and self.subset is not None and idx in self.subset:  # czy podczas treningu prawe wejście nie zostanie rozmazane
            image2 = self.transform3(image2)
        else:
            image2 = self.transform2(image2)

        return (image1, image2), label

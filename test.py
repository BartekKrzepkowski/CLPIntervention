#!/usr/bin/env python3

import os
from torchvision import datasets

dataset_path = os.environ['SVHN_PATH']
    
DOWNLOAD = False
train_dataset = datasets.SVHN(dataset_path, split='train', download=DOWNLOAD)
print(train_dataset)

test_proper_dataset = datasets.SVHN(dataset_path, split='test', download=DOWNLOAD)
print(test_proper_dataset)
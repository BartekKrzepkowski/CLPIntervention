import math
import logging

import torch
from PIL import Image
from torch.utils.data import Dataset


class SplitAndAugmentDataset(Dataset):
    """Split an image into left/right visual fields and transform each field.

    ``subset`` contains training indices whose right field should use
    ``transform3`` instead of ``transform2``. In phase 1 these are the samples
    that keep the proper (non-blurred) right modality.
    """

    def __init__(
        self,
        dataset,
        transform1,
        transform2,
        transform3=None,
        subset=None,
        overlap=0.5,
        is_train=True,
        reverse=True,
    ):
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"overlap must be in [0, 1], got {overlap}")
        if subset is not None and transform3 is None:
            raise ValueError("transform3 is required when subset is provided")

        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        # Membership is checked on every sample, so avoid an O(n) NumPy scan.
        self.subset = None if subset is None else frozenset(int(i) for i in subset)
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
            if torch.rand(()) > 0.5:
                # reverse image horizontally
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        width, height = image.size
        width_ = math.ceil(width * self.with_overlap)
        image1 = image.crop((0, 0, width_, height))
        image2 = image.crop((width-width_, 0, width, height))

        image1 = self.transform1(image1)
        if self.is_train and self.subset is not None and idx in self.subset:
            image2 = self.transform3(image2)
        else:
            image2 = self.transform2(image2)

        return (image1, image2), label

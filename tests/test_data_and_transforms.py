import numpy as np
import pytest
import torch
from PIL import Image

from src.data.datasets_class import SplitAndAugmentDataset
from src.data.transforms_cifar10 import transform_eval_proper as cifar_transform
from src.data.transforms_fmnist import transform_eval_proper as fmnist_transform
from scripts.python_new.get_mean_and_std import streaming_statistics
from src.utils.utils_data import count_classes, get_targets


class ImageDataset:
    def __init__(self):
        self.targets = [0, 1]
        self.classes = ["zero", "one"]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return Image.new("L", (8, 4), color=index), self.targets[index]


class LabelsDataset:
    labels = np.array([2, 0, 1, 2])


def marker(name):
    return lambda image: (name, image.size)


def test_subset_selects_proper_right_transform_only_for_selected_indices():
    dataset = SplitAndAugmentDataset(
        ImageDataset(),
        marker("left"),
        marker("blurred"),
        transform3=marker("proper"),
        subset=np.array([1]),
        overlap=0.0,
        is_train=True,
        reverse=False,
    )

    assert dataset[0][0][1][0] == "blurred"
    assert dataset[1][0][1][0] == "proper"


def test_dataset_validates_overlap_and_subset_contract():
    with pytest.raises(ValueError, match="overlap"):
        SplitAndAugmentDataset(ImageDataset(), marker("a"), marker("b"), overlap=1.1)
    with pytest.raises(ValueError, match="transform3"):
        SplitAndAugmentDataset(ImageDataset(), marker("a"), marker("b"), subset=[0])


def test_target_helpers_support_targets_labels_and_nested_wrappers():
    wrapped = type("Wrapper", (), {"dataset": LabelsDataset()})()
    assert get_targets(wrapped).tolist() == [2, 0, 1, 2]
    assert count_classes(wrapped) == 3
    assert count_classes(SplitAndAugmentDataset(ImageDataset(), marker("a"), marker("b"))) == 2


def test_only_validated_overlap_normalizations_are_accepted():
    cifar_transform(0.125, "left")
    with pytest.raises(ValueError, match="configured values"):
        fmnist_transform(0.125, "left")


def test_streaming_normalization_statistics_use_all_pixels():
    values = torch.tensor([[[[0.0, 1.0]]], [[[1.0, 0.0]]]])
    zeros = torch.zeros_like(values)
    statistics = streaming_statistics([(values, values, zeros)])

    assert statistics["proper_left"] == {"mean": [0.5], "std": [0.5]}
    assert statistics["proper_right"] == {"mean": [0.5], "std": [0.5]}
    assert statistics["blurred_right"] == {"mean": [0.0], "std": [0.0]}

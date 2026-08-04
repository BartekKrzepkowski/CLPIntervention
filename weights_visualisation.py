import argparse
import os

import captum.optim as optimviz
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.prepare import prepare_model
from src.utils.utils_model import load_model_specific_params


def load_checkpoint_model(args):
    model_params = {
        "num_classes": args.num_classes,
        "input_channels": args.input_channels,
        "img_height": args.img_height,
        "img_width": args.img_width,
        "overlap": args.overlap,
        **load_model_specific_params(args.model_name),
    }
    return prepare_model(
        args.model_name, model_params=model_params, model_path=args.checkpoint
    )


class BranchModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def optimize_neuron(model, target, channel, image_size, steps, device):
    image = optimviz.images.NaturalImage(image_size).to(device)
    transforms = torch.nn.Sequential(optimviz.transforms.CenterCrop(image_size))
    loss_fn = optimviz.loss.NeuronActivation(target, channel)
    objective = optimviz.InputOptimization(
        model=model,
        loss_function=loss_fn,
        input_param=image,
        transform=transforms,
    )
    objective.optimize(optimviz.optimization.n_steps(steps, False))
    return image().detach().cpu().numpy()


def save_pair(left_image, right_image, save_path):
    if left_image.shape != right_image.shape:
        raise ValueError("Left and right visualizations must have equal shapes")
    plt.imsave(save_path, np.concatenate((left_image, right_image), axis=1))


def save_grid(images, save_path):
    height, width, channels = images[0].shape
    rows = len(images) // 2
    grid = np.zeros((height * rows, width * 2, channels))
    for index, image in enumerate(images):
        row, column = divmod(index, 2)
        grid[row * height:(row + 1) * height, column * width:(column + 1) * width] = image
    plt.imsave(save_path, grid)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize paired branch neurons from a bimodal checkpoint.")
    parser.add_argument("checkpoint", help="Path to a model state_dict checkpoint")
    parser.add_argument("--model-name", default="mm_resnet")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=16)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(args).to(device).eval()
    left_branch = BranchModel(model.left_branch)
    right_branch = BranchModel(model.right_branch)

    paired_layers = zip(model.left_branch.modules(), model.right_branch.modules())
    layer_index = 0
    for left_layer, right_layer in paired_layers:
        if not isinstance(left_layer, torch.nn.Conv2d):
            continue
        layer_path = os.path.join(args.output_dir, args.model_name, str(layer_index))
        os.makedirs(layer_path, exist_ok=True)
        images = []
        for channel in range(left_layer.out_channels):
            left_image = optimize_neuron(
                left_branch, left_layer, channel, (args.image_size, args.image_size), args.steps, device
            ).squeeze().transpose(1, 2, 0)
            right_image = optimize_neuron(
                right_branch, right_layer, channel, (args.image_size, args.image_size), args.steps, device
            ).squeeze().transpose(1, 2, 0)
            images.extend((left_image, right_image))
            save_pair(left_image, right_image, os.path.join(layer_path, f"{channel}_neurons.png"))
        save_grid(images, os.path.join(layer_path, "grid_of_neurons.png"))
        layer_index += 1


if __name__ == "__main__":
    main()

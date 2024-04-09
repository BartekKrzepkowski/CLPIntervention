import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import captum
import captum.optim as optimviz
import torch
import torchvision
from src.utils.utils_model import load_model_specific_params
from src.utils.prepare import prepare_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_name, checkpoint_path):
    model_params = load_model_specific_params(model_name)
    model_params = {
        'num_classes': 10,
        'input_channels': 3,
        'img_height': 32,
        'img_width': 32,
        'overlap': 00,
        **model_params
        }
        
    model = prepare_model(model_name, model_params=model_params)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return model

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

LossFunction = Callable[[Dict[torch.nn.Module, Optional[torch.Tensor]]], torch.Tensor]


def show(
    x: torch.Tensor, figsize: Optional[Tuple[int, int]] = None, scale: float = 255.0
) -> None:
    assert x.dim() == 3 or x.dim() == 4
    x = x[0] if x.dim() == 4 else x
    x = x.cpu().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


def vis_neuron_large(
    model: torch.nn.Module, target: torch.nn.Module, channel: int, figsize: Tuple[int, int], steps: int
) -> None:
    image = optimviz.images.NaturalImage(figsize).to(device)
    transforms = torch.nn.Sequential(
        optimviz.transforms.CenterCrop(figsize),
    )
    loss_fn = optimviz.loss.NeuronActivation(target, channel)
    obj = optimviz.InputOptimization(model=model, loss_function=loss_fn, input_param=image, transform=transforms)
    history = obj.optimize(optimviz.optimization.n_steps(steps, False))
    return image().detach().cpu().numpy()
    
    
    
class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        
    def forward(self, x):
        x = self.model(x)
        return x
    
def save_imgs(img1, img2, save_path):
    """
    Funkcja do łączenia dwóch zdjęć w formacie RGB reprezentowanych jako array numpy
    i zapisywania ich jako jednego zdjęcia w wskazanym miejscu.

    Parametry:
    zdjecie1 (numpy array): Pierwsze zdjęcie w formacie RGB.
    zdjecie2 (numpy array): Drugie zdjęcie w formacie RGB.
    sciezka_zapisu (str): Ścieżka do zapisu połączonego zdjęcia.
    """
    # Sprawdzenie czy oba zdjęcia mają taki sam rozmiar
    if img1.shape != img2.shape:
        raise ValueError("Zdjęcia muszą mieć ten sam rozmiar")

    # Połączenie obu zdjęć w jedno, umieszczając je obok siebie
    polaczone_zdjecie = np.concatenate((img1, img2), axis=1)

    # Zapisanie połączonych zdjęć do pliku
    plt.imsave(save_path, polaczone_zdjecie)
    
    
def save_grid(imgs, save_path, wymiar_gridu=(100, 2)):
    """
    Funkcja do łączenia wielu zdjęć w formacie RGB reprezentowanych jako array numpy
    i zapisywania ich jako jednego zdjęcia w wskazanym miejscu, ułożonych w grid.

    Parametry:
    zdjecia (list of numpy arrays): Lista zdjęć w formacie RGB.
    sciezka_zapisu (str): Ścieżka do zapisu zdjęcia grid.
    wymiar_gridu (tuple): Wymiary gridu, domyślnie (100, 2).
    """

    # Rozmiar jednego zdjęcia
    wysokosc, szerokosc, kanaly = imgs[0].shape

    # Stworzenie pustego obrazu o odpowiednich wymiarach
    grid_wysokosc = wysokosc * wymiar_gridu[0]
    grid_szerokosc = szerokosc * wymiar_gridu[1]
    grid_zdjecie = np.zeros((grid_wysokosc, grid_szerokosc, kanaly))

    # Wypełnienie gridu zdjęciami
    for i, zdjecie in enumerate(imgs):
        row = i // wymiar_gridu[1]  # Indeks wiersza w gridzie
        col = i % wymiar_gridu[1]   # Indeks kolumny w gridzie
        grid_zdjecie[row*wysokosc:(row+1)*wysokosc, col*szerokosc:(col+1)*szerokosc, :] = zdjecie

    # Zapisanie gridu zdjęć do pliku
    plt.imsave(save_path, grid_zdjecie)

    
    
if __name__ == '__main__':
    
    model_name = 'mm_resnet'
    checkpoint_path = '/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2/all_at_once, training with phase1=0, phase2=180, phase3=0, phase4=0, mm_cifar10, mm_resnet, sgd, overlap=0.0_lr=0.5_wd=0.0_lambda=1.0/2023-12-22_03-19-13/checkpoints/model_step_epoch_180.pth'

    model_clean = load_model(model_name, checkpoint_path)
    model_clean = model_clean.to(device).eval()
    
    size = 128
    figsize = (size, size)
    steps = 128
    
    left_branch = Model(model_clean.left_branch)
    right_branch = Model(model_clean.right_branch)
    
    layer_name = 0
    for layer_left, layer_right in zip(model_clean.left_branch.modules(), model_clean.right_branch.modules()):
        if not isinstance(layer_left, torch.nn.Conv2d):
            continue
        base_path = f'features/{model_name}/{layer_name}'
        imgs = []
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for i in range(layer_left.weight.shape[0]):
            save_path = f'{base_path}/{i}_kernels.png'
            img1 = vis_neuron_large(left_branch, layer_left, i, figsize, steps).squeeze().transpose(1,2,0)
            img2 = vis_neuron_large(right_branch, layer_right, i, figsize, steps).squeeze().transpose(1,2,0)
            imgs.append(img1)
            imgs.append(img2)
            save_imgs(img1, img2, save_path)
        save_path = f'{base_path}/grid_of_kernels.png'
        save_grid(imgs, save_path, wymiar_gridu=(len(imgs)//2, 2))
        layer_name += 1
        
    
    
    
    
    
    
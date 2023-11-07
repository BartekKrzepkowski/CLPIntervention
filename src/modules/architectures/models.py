from typing import List, Dict, Any

import torch

from src.modules.architectures import aux_modules
from src.utils.utils_model import infer_flatten_dim
from src.utils import common


class MLP(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2), common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class MLP_scaled(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(aux_modules.PreAct(hidden_dim1), torch.nn.Linear(hidden_dim1, hidden_dim2), common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Sequential(aux_modules.PreAct(layers_dim[-2]), torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

class MLPwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, norm_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                common.NORM_LAYER_NAME_MAP[norm_name](hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    
class SimpleCNNwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x


class SimpleCNNwithDropout(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Dropout(0.05),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    
class SimpleCNNwithNormandDropout(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               torch.nn.Dropout(0.05),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    
class DualSimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any], wheter_concate: bool = False,
                 pre_mlp_depth: int = 1, eps: float = 1e-5, overlap: float = 0.0, num_features: int = 1):
        from math import ceil
        super().__init__()
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        self.net1 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        self.net2 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        
        # x1 = torch.randn(1, 1, 28, ceil(28 * (overlap / 2 + 0.5)))
        x1 = torch.randn(1, 3, 32, ceil(32 * (overlap / 2 + 0.5))) if num_features == 3 else torch.randn(1, 1, 28, ceil(28 * (overlap / 2 + 0.5)))
        for block in self.net1:
            x1 = block(x1)
        _, self.channels_out, self.height, self.width = x1.shape
        pre_mlp_channels = self.channels_out * self.scaling_factor
        flatten_dim = int(self.height * self.width * pre_mlp_channels)
        
        self.net3 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(pre_mlp_channels, pre_mlp_channels, 3, padding=1),
                                torch.nn.BatchNorm2d(pre_mlp_channels),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(pre_mlp_channels, pre_mlp_channels, 3, padding=1),
                                torch.nn.BatchNorm2d(pre_mlp_channels),
                                common.ACT_NAME_MAP[activation_name](),
                                # torch.nn.MaxPool2d(2, 2)
                            )
            for _ in range(pre_mlp_depth)
        ])
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))
        
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            for block in self.net1:
                x1 = block(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn((x1.size(0), self.channels_out, self.height, self.width), device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros((x1.size(0), self.channels_out, self.height, self.width), device=x1.device)
            else:
                raise ValueError("Invalid left branch intervention")
        
        if enable_right_branch:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn_like(x2, device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros_like(x2, device=x2.device)
                
            for block in self.net2:
                x2 = block(x2)
        else:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn((x2.size(0), self.channels_out, self.height, self.width), device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros((x2.size(0), self.channels_out, self.height, self.width), device=x2.device)
            else:
                raise ValueError("Invalid right branch intervention")
                
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        for block in self.net3:
            y = block(y)
        y = y.flatten(start_dim=1)
        y = self.final_layer(y)
        return y
    
    
class Net(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.net1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(32, 32, 3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2)
                            )
        self.net2 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(32, 32, 3, padding=1),
                                torch.nn.BatchNorm2d(32),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2)
                            )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
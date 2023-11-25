from typing import List, Dict, Any

import torch

from src.utils import common
from src.utils.utils_model import infer_dims_from_blocks

    
class MMSimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str = 'relu', wheter_concate: bool = False,
                 pre_mlp_depth: int = 1, eps: float = 1e-5, overlap: float = 0.0, num_features: int = 1):
        from math import ceil
        super().__init__()
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        
        self.left_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 5, padding=2, stride=2),
                                torch.nn.BatchNorm2d(layer_dim2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        self.right_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 5, padding=2, stride=2),
                                torch.nn.BatchNorm2d(layer_dim2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        
        z = torch.randn(1, 3, 32, ceil(32 * (overlap / 2 + 0.5))) if num_features == 3 else torch.randn(1, 1, 28, ceil(28 * (overlap / 2 + 0.5)))
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        pre_mlp = [pre_mlp_channels for i in range(pre_mlp_depth + 1)]
        flatten_dim = int(self.height * self.width * pre_mlp[-1])
        
        self.main_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(pre_mlp[i], pre_mlp[i+1], 3, padding=1),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(pre_mlp[i+1], pre_mlp[i+1], 3, padding=1),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(pre_mlp[i+1], pre_mlp[i+1], 3, padding=1),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                common.ACT_NAME_MAP[activation_name]()
                            )
            for i in range(pre_mlp_depth)
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
                
            for block in self.left_branch:
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
                
            for block in self.right_branch:
                x2 = block(x2)
        else:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn((x2.size(0), self.channels_out, self.height, self.width), device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros((x2.size(0), self.channels_out, self.height, self.width), device=x2.device)
            else:
                raise ValueError("Invalid right branch intervention")
                
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        for block in self.main_branch:
            y = block(y)
        y = y.flatten(start_dim=1)
        y = self.final_layer(y)
        return y
    

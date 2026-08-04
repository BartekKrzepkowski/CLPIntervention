from typing import List

import torch

ACT_NAME_MAP = {
    "gelu": torch.nn.GELU,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}

from src.utils.utils_model import infer_dims_from_blocks

    
class MMSimpleCNN(torch.nn.Module):
    def __init__(self, num_classes, input_channels, img_height, img_width, overlap, hidden_layers_dim: List[int],
                 activation_name: str = 'relu', wheter_concate: bool = False, pre_mlp_depth: int = 1,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scaling_factor = 2 if wheter_concate else 1
        layers_dim = [input_channels] + hidden_layers_dim + [num_classes]
        
        self.left_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 5, padding=2, stride=2, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        self.right_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 5, padding=2, stride=2, bias=False),
                                torch.nn.BatchNorm2d(layer_dim2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        
        z = torch.zeros(1, input_channels, img_height, img_width)
        self.channels_out, self.height, self.width, pre_mlp_channels = infer_dims_from_blocks(self.left_branch, z, scaling_factor=self.scaling_factor)
        pre_mlp = [pre_mlp_channels for i in range(pre_mlp_depth + 1)]
        flatten_dim = int(self.height * self.width * pre_mlp[-1])
        
        self.main_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(pre_mlp[i], pre_mlp[i+1], 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(pre_mlp[i+1], pre_mlp[i+1], 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(pre_mlp[i+1], pre_mlp[i+1], 3, padding=1, bias=False),
                                torch.nn.BatchNorm2d(pre_mlp[i+1]),
                                ACT_NAME_MAP[activation_name]()
                            )
            for i in range(pre_mlp_depth)
        ])
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2], bias=False),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))
        
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True, return_features=False):
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
                x1 = x1.new_empty((x1.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = x1.new_zeros((x1.size(0), self.channels_out, self.height, self.width))
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
                x2 = x2.new_empty((x2.size(0), self.channels_out, self.height, self.width)).normal_() * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = x2.new_zeros((x2.size(0), self.channels_out, self.height, self.width))
            else:
                raise ValueError("Invalid right branch intervention")
                
        features_left, features_right = x1, x2
        y = torch.cat((x1, x2), dim=1) if self.scaling_factor == 2 else x1 + x2
        for block in self.main_branch:
            y = block(y)
        y = y.flatten(start_dim=1)
        y = self.final_layer(y)
        return (y, features_left, features_right) if return_features else y
    

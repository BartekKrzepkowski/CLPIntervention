from typing import List

import torch

ACT_NAME_MAP = {
    "gelu": torch.nn.GELU,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "sigmoid": torch.nn.Sigmoid,
    "tanh": torch.nn.Tanh,
}



class MMMLPwithNorm(torch.nn.Module):
    def __init__(self, num_classes, input_channels, img_height, img_width, overlap, hidden_layers_dim: List[int], activation_name: str, scaling_factor: int = 1, eps: float = 1e-5):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.eps = eps
        hidden_layers_dim = [input_channels * img_height * img_width] + hidden_layers_dim + [num_classes]
        layers_dim1 = hidden_layers_dim[:len(hidden_layers_dim) // 2]
        layers_dim2 = hidden_layers_dim[len(hidden_layers_dim) // 2:]
        branch_output_dim = layers_dim1[-1]
        layers_dim2[0] = branch_output_dim * scaling_factor
        self.left_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim1[:-1], layers_dim1[1:])
        ])
        self.left_branch = torch.nn.Sequential(*self.left_branch)
        self.right_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim1[:-1], layers_dim1[1:])
        ])
        self.right_branch = torch.nn.Sequential(*self.right_branch)
        self.hidden_dim = branch_output_dim
        self.main_branch = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim2[:-2], layers_dim2[1:-1])
        ])
        self.main_branch = torch.nn.Sequential(*self.main_branch)
        self.fc = torch.nn.Linear(hidden_layers_dim[-2], hidden_layers_dim[-1])
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True, return_features=False):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        
        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            x1 = self.left_branch(x1)
        else:
            if left_branch_intervention == "occlusion":
                x1 = x1.new_empty((x1.size(0), self.hidden_dim)).normal_() * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = x1.new_zeros((x1.size(0), self.hidden_dim))
            else:
                raise ValueError("Invalid left branch intervention")
        
        if enable_right_branch:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn_like(x2, device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros_like(x2, device=x2.device)
                
            x2 = self.right_branch(x2)
        else:
            if right_branch_intervention == "occlusion":
                x2 = x2.new_empty((x2.size(0), self.hidden_dim)).normal_() * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = x2.new_zeros((x2.size(0), self.hidden_dim))
            else:
                raise ValueError("Invalid right branch intervention")
            
        features_left, features_right = x1, x2
        y = torch.cat((x1, x2), dim=-1) if self.scaling_factor == 2 else x1 + x2
        y = self.main_branch(y)
        y = self.fc(y)
        return (y, features_left, features_right) if return_features else y
from typing import List

import torch

from src.utils import common



class MMMLPwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, scaling_factor: int = 1, eps: float = 0.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.eps = eps
        layers_dim1 = layers_dim[:len(layers_dim) // 2]
        layers_dim2 = layers_dim[len(layers_dim) // 2:]
        self.net1 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim1[:-1], layers_dim1[1:])
        ])
        self.net1 = torch.nn.Sequential(*self.net1)
        self.net2 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim1[:-1], layers_dim1[1:])
        ])
        self.net2 = torch.nn.Sequential(*self.net2)
        self.net3 = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim2[:-2], layers_dim2[1:-1])
        ])
        self.net3 = torch.nn.Sequential(*self.net3)
        self.fc = torch.nn.Linear(layers_dim[-2], layers_dim[-1])
        
    
    def forward(self, x1, x2, left_branch_intervention=None, right_branch_intervention=None, enable_left_branch=True, enable_right_branch=True):
        assert left_branch_intervention is None or right_branch_intervention is None, "At least one branchnet should be left intact"
        assert enable_left_branch or enable_right_branch, "At least one branchnet should be enabled"
        
        x1 = x1.flatten(start_dim=1)
        x2 = x2.flatten(start_dim=1)
        
        if enable_left_branch:
            if left_branch_intervention == "occlusion":
                x1 = torch.randn_like(x1, device=x1.device) * self.eps
            elif left_branch_intervention == "deactivation":
                x1 = torch.zeros_like(x1, device=x1.device)
                
            x1 = self.net1(x1)
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
                
            x2 = self.net2(x2)
        else:
            if right_branch_intervention == "occlusion":
                x2 = torch.randn((x2.size(0), self.channels_out, self.height, self.width), device=x2.device) * self.eps
            elif right_branch_intervention == "deactivation":
                x2 = torch.zeros((x2.size(0), self.channels_out, self.height, self.width), device=x2.device)
            else:
                raise ValueError("Invalid right branch intervention")
            
        y = torch.cat((x1, x2), dim=-1) if self.scaling_factor == 2 else x1 + x2
        y = self.net3(y)
        y = self.fc(y)
        return y
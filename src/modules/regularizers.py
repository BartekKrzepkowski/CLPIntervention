from collections import defaultdict

import torch
from torch.distributions import Categorical

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
from src.utils.utils_regularizers import get_desired_parameter_names


class FisherPenaly(torch.nn.Module):
    def __init__(self, model, criterion, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.labels = torch.arange(num_classes).to(next(model.parameters()).device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)

    def forward(self, y_pred):
        y_sampled = Categorical(logits=y_pred).sample()
        loss = self.criterion(y_pred, y_sampled)
        params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=True)
        traces = defaultdict(float)
        overall_trace = 0.0
        # najlepiej rozdzielić po module
        for param_name, gr in zip(params_names, grads):
            if gr is not None:
                trace_p = (gr**2).sum()
                traces[param_name] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p
        return overall_trace, traces
    
    
    
class BalancePenaly(torch.nn.Module):
    def __init__(self, model, criterion, num_classes):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.labels = torch.arange(num_classes).to(next(model.parameters()).device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)

    def forward(self, y_pred):
        eps = torch.finfo(torch.float32).eps
        y_sampled = Categorical(logits=y_pred).sample()
        loss = self.criterion(y_pred, y_sampled)
        named_params = {n: p for n, p in self.model.named_parameters() if n in self.penalized_parameter_names and p.requires_grad}
        # params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        params_names_left, params_left = zip(*[(n, p) for n, p in named_params.items() if 'left_branch' in n])
        params_names_right, params_right = zip(*[(n, p) for n, p in named_params.items() if 'right_branch' in n])
        traces = defaultdict(float)
        grads_left = torch.autograd.grad(
            loss,
            params_left,
            retain_graph=True,
            create_graph=True)
        overall_trace_left = 0.0
        for param_name, gr in zip(params_names_left, grads_left):
            if gr is not None:
                trace_p = (gr**2).sum()
                traces[f'balance_penalty/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace_left += trace_p
                    
        
        grads_right = torch.autograd.grad(
            loss,
            params_right,
            retain_graph=True,
            create_graph=True)
        overall_trace_right = 0.0
        for param_name, gr in zip(params_names_right, grads_right):
            if gr is not None:
                trace_p = (gr**2).sum()
                traces[f'balance_penalty/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace_right += trace_p
                    
        mean_ratio = (overall_trace_left / (overall_trace_right + eps) + overall_trace_right / (overall_trace_left + eps)) / 2
        sum_of_fims = overall_trace_left + overall_trace_right
        return mean_ratio, sum_of_fims, traces


        

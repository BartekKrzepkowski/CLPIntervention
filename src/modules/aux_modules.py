from collections import defaultdict
from copy import deepcopy

import torch
from torch.distributions import Categorical

from src.utils import prepare
from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES

from torch.func import functional_call, vmap, grad


class TraceFIM(torch.nn.Module): #OverheadPrevention
    def __init__(self, held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out['proper_x_left']
        self.held_out_proper_x_right = held_out['proper_x_right']
        self.held_out_blurred_x_right = held_out['blurred_x_right']
        
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(self.grad_and_trace, in_dims=(None, None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("\n penalized parameter names TFIM: ", self.penalized_parameter_names, '\n')
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, config, sample):
        batch0 = sample[0].unsqueeze(0)
        batch1 = sample[1].unsqueeze(0)
        kwargs = {'left_branch_intervention': config.extra['left_branch_intervention'],
                  'right_branch_intervention': config.extra['right_branch_intervention'],
                  'enable_left_branch': config.extra['enable_left_branch'],
                  'enable_right_branch': config.extra['enable_right_branch']}
        y_pred = functional_call(self.model, (params, buffers), (batch0, batch1), kwargs=kwargs)
        # y_sampled = Categorical(logits=y_pred).sample()
        idx_sampled = torch.nn.functional.softmax(y_pred, dim=1).multinomial(1)
        loss = self.criterion(y_pred, self.labels[idx_sampled].long().squeeze(-1))
        return loss
    
    def grad_and_trace(self, params, buffers, config, sample):
        sample_traces = {}
        sample_grads = grad(self.compute_loss, has_aux=False)(params, buffers, config, sample)
        for param_name in sample_grads:
            gr = sample_grads[param_name]
            if gr is not None:
                trace_p = (torch.pow(gr, 2)).sum()
                sample_traces[param_name] = trace_p
        return sample_traces

    def forward(self, global_step, config, kind):
        self.model.eval()
        x_true1 = self.held_out_proper_x_left.to(self.device)
        x_true2 = self.held_out_proper_x_right.to(self.device) if kind == 'proper' else self.held_out_blurred_x_right.to(self.device)
        
        params = {n: p.detach() for n, p in self.model.named_parameters() if n in self.penalized_parameter_names and p.requires_grad}
        params1 = {n: p for n, p in params.items() if 'left_branch' in n}
        params2 = {n: p for n, p in params.items() if 'right_branch' in n}
        params3 = {n: p for n, p in params.items() if 'main_branch' in n}
        buffers = {}
        ft_per_sample_grads1 = self.ft_criterion(params1, buffers, config, (x_true1, x_true2))
        ft_per_sample_grads1 = {k1: v.detach().data for k1, v in ft_per_sample_grads1.items()}
        ft_per_sample_grads2 = self.ft_criterion(params2, buffers, config, (x_true1, x_true2))
        ft_per_sample_grads2 = {k1: v.detach().data for k1, v in ft_per_sample_grads2.items()}
        ft_per_sample_grads3 = {}
        # ft_per_sample_grads3 = self.ft_criterion(params3, buffers, config, (x_true1, x_true2))
        # ft_per_sample_grads3 = {k1: v.detach().data for k1, v in ft_per_sample_grads3.items()}
        ft_per_sample_grads = ft_per_sample_grads1 | ft_per_sample_grads2 | ft_per_sample_grads3
        params_names1 = [n for n, _ in params1.items()]
        params_names2 = [n for n, _ in params2.items()]
        params_names3 = [n for n, _ in params3.items()]
        evaluators = defaultdict(float)
        overall_trace = 0.0
        overall_trace1_bias = 0.0
        overall_trace1_weight = 0.0
        overall_trace2_bias = 0.0
        overall_trace2_weight = 0.0
        overall_trace3_bias = 0.0
        overall_trace3_weight = 0.0
        for param_name in ft_per_sample_grads:
            trace_p = ft_per_sample_grads[param_name].mean()          
            evaluators[f'trace_fim_{kind}/{param_name}'] += trace_p.item()
            if param_name in self.penalized_parameter_names:
                # overall_trace += trace_p.item()
                if param_name in params_names1:
                    if 'bias' in param_name:
                        overall_trace1_bias += trace_p.item()
                    elif 'weight' in param_name:
                        overall_trace1_weight += trace_p.item()
                    else:
                        raise ValueError("The parameters are neither biases nor weights.")
                elif param_name in params_names2:
                    if 'bias' in param_name:
                        overall_trace2_bias += trace_p.item()
                    elif 'weight' in param_name:
                        overall_trace2_weight += trace_p.item()
                    else:
                        raise ValueError("The parameters are neither biases nor weights.")
                elif param_name in params_names3:
                    if 'bias' in param_name:
                        overall_trace3_bias += trace_p.item()
                    elif 'weight' in param_name:
                        overall_trace3_weight += trace_p.item()
                    else:
                        raise ValueError("The parameters are neither biases nor weights.")
        
        evaluators[f'trace_fim_overall/{kind}_trace_bias'] = overall_trace1_bias + overall_trace2_bias + overall_trace3_bias
        evaluators[f'trace_fim_overall/{kind}_trace_weight'] = overall_trace1_weight + overall_trace2_weight + overall_trace3_weight
        evaluators[f'trace_fim_overall/{kind}_trace'] = evaluators[f'trace_fim_overall/{kind}_trace_bias'] + evaluators[f'trace_fim_overall/{kind}_trace_weight']
        
        evaluators[f'trace_fim_overall/{kind}_trace1_bias'] = overall_trace1_bias
        evaluators[f'trace_fim_overall/{kind}_trace1_weight'] = overall_trace1_weight
        evaluators[f'trace_fim_overall/{kind}_trace1'] = overall_trace1_bias + overall_trace1_weight
        
        evaluators[f'trace_fim_overall/{kind}_trace2_bias'] = overall_trace2_bias
        evaluators[f'trace_fim_overall/{kind}_trace2_weight'] = overall_trace2_weight
        evaluators[f'trace_fim_overall/{kind}_trace2'] = overall_trace2_bias + overall_trace2_weight
        
        evaluators[f'trace_fim_overall/{kind}_trace3_bias'] = overall_trace3_bias
        evaluators[f'trace_fim_overall/{kind}_trace3_weight'] = overall_trace3_weight
        evaluators[f'trace_fim_overall/{kind}_trace3'] = overall_trace3_bias + overall_trace3_weight
        
        evaluators[f'trace_fim_overall/{kind}_ratio_left_to_right_bias'] = overall_trace1_bias / (overall_trace2_bias + 1e-10)
        evaluators[f'trace_fim_overall/{kind}_ratio_left_to_right_weight'] = overall_trace1_weight / (overall_trace2_weight + 1e-10)
        evaluators[f'trace_fim_overall/{kind}_ratio_left_to_right'] = (overall_trace1_bias + overall_trace1_weight) / (overall_trace2_bias + overall_trace2_weight + 1e-10)
        # evaluators[f'trace_fim_{kind}/overall_ratio_1_to_3'] = overall_trace1 / (overall_trace3 + 1e-10)
        # evaluators[f'trace_fim_{kind}/overall_ratio_2_to_3'] = overall_trace2 / (overall_trace3 + 1e-10)
        evaluators['steps/trace_fim'] = global_step
        self.model.train()
        self.logger.log_scalars(evaluators, global_step)    
        
        
class TraceFIM__(torch.nn.Module):
    def __init__(self, held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out['proper_x_left']
        self.held_out_proper_x_right = held_out['proper_x_right']
        self.held_out_blurred_x_right = held_out['blurred_x_right']
        
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(self.grad_and_trace, in_dims=(None, None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, config, sample):
        batch0 = sample[0].unsqueeze(0)
        batch1 = sample[1].unsqueeze(0)
        kwargs = {'left_branch_intervention': config.extra['left_branch_intervention'],
                  'right_branch_intervention': config.extra['right_branch_intervention'],
                  'enable_left_branch': config.extra['enable_left_branch'],
                  'enable_right_branch': config.extra['enable_right_branch']}
        y_pred = functional_call(self.model, (params, buffers), (batch0, batch1), kwargs=kwargs)
        # y_sampled = Categorical(logits=y_pred).sample()
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze(-1)
        loss = self.criterion(y_pred, y_sampled)
        return loss
    
    def grad_and_trace(self, params, buffers, config, sample):
        sample_traces = {}
        sample_grads = grad(self.compute_loss, has_aux=False)(params, buffers, config, sample)
        for param_name in sample_grads:
            gr = sample_grads[param_name]
            if gr is not None:
                trace_p = (torch.pow(gr, 2)).sum()
                sample_traces[param_name] = trace_p
        return sample_traces

    def forward(self, step, config, kind):
        self.model.eval()
        x_true1 = self.held_out_proper_x_left.to(self.device)
        x_true2 = self.held_out_proper_x_right.to(self.device) if kind == 'proper' else self.held_out_blurred_x_right.to(self.device)
        
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, config, (x_true1, x_true2))
        ft_per_sample_grads = {k1: v.detach().data for k1, v in ft_per_sample_grads.items()}
        
        params_names1 = [n for n, _ in params.items() if 'net1' in n]
        params_names2 = [n for n, _ in params.items() if 'net2' in n]
        params_names3 = [n for n, _ in params.items() if 'net3' in n]
        params_nb1 = sum([p.numel() for n, p in params.items() if 'net1' in n])
        params_nb2 = sum([p.numel() for n, p in params.items() if 'net2' in n])
        params_nb3 = sum([p.numel() for n, p in params.items() if 'net3' in n])
        evaluators = defaultdict(float)
        overall_trace = 0.0
        overall_trace1 = 0.0
        overall_trace2 = 0.0
        overall_trace3 = 0.0
        overall_trace1_normalized = 0.0
        overall_trace2_normalized = 0.0
        overall_trace3_normalized = 0.0
        for param_name in ft_per_sample_grads:
            trace_p = ft_per_sample_grads[param_name].mean()          
            evaluators[f'trace_fim_{kind}/{param_name}'] += trace_p.item()
            if param_name in self.penalized_parameter_names:
                overall_trace += trace_p.item()
                params_nb = sum([p.numel() for n, p in params.items() if n == param_name])
                if param_name in params_names1:
                    overall_trace1 += trace_p.item()
                    overall_trace1_normalized += trace_p.item() / params_nb
                elif param_name in params_names2:
                    overall_trace2 += trace_p.item()
                    overall_trace2_normalized += trace_p.item() / params_nb
                elif param_name in params_names3:
                    overall_trace3 += trace_p.item()
                    overall_trace3_normalized += trace_p.item() / params_nb
        
        evaluators[f'trace_fim_{kind}/overall_trace'] = overall_trace
        evaluators[f'trace_fim_{kind}/overall_trace1'] = overall_trace1
        evaluators[f'trace_fim_{kind}/overall_trace2'] = overall_trace2
        evaluators[f'trace_fim_{kind}/overall_trace3'] = overall_trace3
        evaluators[f'trace_fim_{kind}/overall_ratio_1_to_2'] = overall_trace1 / (overall_trace2 + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_1_to_3'] = overall_trace1 / (overall_trace3 + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_2_to_3'] = overall_trace2 / (overall_trace3 + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_1_to_3_normalized_local'] = overall_trace1_normalized / (overall_trace3_normalized + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_2_to_3_normalized_local'] = overall_trace2_normalized / (overall_trace3_normalized + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_1_to_3_normalized_global'] = (overall_trace1 / params_nb1) / (overall_trace3 / params_nb3 + 1e-10)
        evaluators[f'trace_fim_{kind}/overall_ratio_2_to_3_normalized_global'] = (overall_trace2 / params_nb2) / (overall_trace3 / params_nb3 + 1e-10)
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)    
        
        
class TraceFIM_(torch.nn.Module):
    def __init__(self, held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out['proper_x_left']
        self.held_out_proper_x_right = held_out['proper_x_right']
        self.held_out_blurred_x_right = held_out['blurred_x_right']
        
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=False), in_dims=(None, None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, config, sample):
        batch0 = sample[0].unsqueeze(0)
        batch1 = sample[1].unsqueeze(0)
        kwargs = {'left_branch_intervention': config.extra['left_branch_intervention'],
                  'right_branch_intervention': config.extra['right_branch_intervention'],
                  'enable_left_branch': config.extra['enable_left_branch'],
                  'enable_right_branch': config.extra['enable_right_branch']}
        y_pred = functional_call(self.model, (params, buffers), (batch0, batch1), kwargs=kwargs)
        # y_sampled = Categorical(logits=y_pred).sample()
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze(-1)
        loss = self.criterion(y_pred, y_sampled)
        return loss

    def forward(self, step, config, kind):
        self.model.eval()
        x_true1 = self.held_out_proper_x_left.to(self.device)
        x_true2 = self.held_out_proper_x_right.to(self.device) if kind == 'proper' else self.held_out_blurred_x_right.to(self.device)
        
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, config, (x_true1, x_true2))
        ft_per_sample_grads = {k1: v.detach().data for k1, v in ft_per_sample_grads.items()}
        
        params_names1 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net1' in n]
        params_names2 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net2' in n]
        params_names3 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net3' in n]
        evaluators = defaultdict(float)
        overall_trace = 0.0
        overall_trace1 = 0.0
        overall_trace2 = 0.0
        overall_trace3 = 0.0
        for param_name in ft_per_sample_grads:
            gr = ft_per_sample_grads[param_name]
            if gr is not None:
                trace_p = (gr**2).sum() / gr.size(0)
                evaluators[f'trace_fim_{kind}/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p.item()
                    if param_name in params_names1:
                        overall_trace1 += trace_p.item()
                    elif param_name in params_names2:
                        overall_trace2 += trace_p.item()
                    elif param_name in params_names3:
                        overall_trace3 += trace_p.item()
        
        evaluators[f'trace_fim_{kind}/overall_trace'] = overall_trace
        evaluators[f'trace_fim_{kind}/overall_trace1'] = overall_trace1
        evaluators[f'trace_fim_{kind}/overall_trace2'] = overall_trace2
        evaluators[f'trace_fim_{kind}/overall_trace3'] = overall_trace3
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)      
        
        
class TraceFIMB(torch.nn.Module):
    def __init__(self, held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out['proper_x_left']
        self.held_out_proper_x_right = held_out['proper_x_right']
        self.held_out_blurred_x_right = held_out['blurred_x_right']
        
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.labels = torch.arange(num_classes).to(self.device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.logger = None

    def forward(self, step, config, kind):
        x_true1 = self.held_out_proper_x_left.to(self.device)
        x_true2 = self.held_out_proper_x_right.to(self.device) if kind == 'proper' else self.held_out_blurred_x_right.to(self.device)
        y_pred = self.model(x_true1, x_true2, 
                            left_branch_intervention=config.extra['left_branch_intervention'],
                            right_branch_intervention=config.extra['right_branch_intervention'],
                            enable_left_branch=config.extra['enable_left_branch'],
                            enable_right_branch=config.extra['enable_right_branch'])
        y_sampled = Categorical(logits=y_pred).sample()
        loss = self.criterion(y_pred, y_sampled)
        params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        params_names1 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net1' in n]
        params_names2 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net2' in n]
        params_names3 = [n for n, p in self.model.named_parameters() if p.requires_grad and 'net3' in n]
        grads = torch.autograd.grad(
            loss,
            params,
            allow_unused=True)
        evaluators = defaultdict(float)
        overall_trace = 0.0
        overall_trace1 = 0.0
        overall_trace2 = 0.0
        overall_trace3 = 0.0
        for param_name, gr in zip(params_names, grads):
            if gr is not None:
                trace_p = (gr**2).sum()
                evaluators[f'trace_fim_{kind}/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p.item()
                    if param_name in params_names1:
                        overall_trace1 += trace_p.item()
                    elif param_name in params_names2:
                        overall_trace2 += trace_p.item()
                    elif param_name in params_names3:
                        overall_trace3 += trace_p.item()
        
        evaluators[f'trace_fim_{kind}/overall_trace'] = overall_trace # czy nie trzeba tego przypadkiem mnożyć przez wielkość batcha?
        evaluators[f'trace_fim_{kind}/overall_trace1'] = overall_trace1
        evaluators[f'trace_fim_{kind}/overall_trace2'] = overall_trace2
        evaluators[f'trace_fim_{kind}/overall_trace3'] = overall_trace3
        evaluators['steps/trace_fim'] = step
        self.logger.log_scalars(evaluators, step)       


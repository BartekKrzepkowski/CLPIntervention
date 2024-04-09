from collections import defaultdict
from copy import deepcopy

import torch
from torch.distributions import Categorical
from torch.func import functional_call, vmap, grad

from src.modules.aux_modules_collapse import variance_eucl
from src.utils import prepare
from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES

class TraceFIM(torch.nn.Module): #OverheadPrevention
    def __init__(self, held_out, model, num_classes, postfix, m_sampling=1):
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
        self.postfix = postfix
        self.m_sampling = m_sampling
        
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
        evaluators = defaultdict(float)
        overall_trace = 0.0
        overall_trace1_bias = 0.0
        overall_trace1_weight = 0.0
        overall_trace2_bias = 0.0
        overall_trace2_weight = 0.0
        overall_trace3_bias = 0.0
        overall_trace3_weight = 0.0
        for _ in range(self.m_sampling):
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
            for param_name in ft_per_sample_grads:
                trace_p = ft_per_sample_grads[param_name].mean()          
                evaluators[f'trace_fim_{self.postfix}_{kind}/{param_name}'] += trace_p.item() / self.m_sampling
                if param_name in self.penalized_parameter_names:
                    # overall_trace += trace_p.item()
                    if param_name in params_names1:
                        if 'bias' in param_name:
                            overall_trace1_bias += trace_p.item() / self.m_sampling
                        elif 'weight' in param_name:
                            overall_trace1_weight += trace_p.item() / self.m_sampling
                        else:
                            raise ValueError("The parameters are neither biases nor weights.")
                    elif param_name in params_names2:
                        if 'bias' in param_name:
                            overall_trace2_bias += trace_p.item() / self.m_sampling
                        elif 'weight' in param_name:
                            overall_trace2_weight += trace_p.item() / self.m_sampling
                        else:
                            raise ValueError("The parameters are neither biases nor weights.")
                    elif param_name in params_names3:
                        if 'bias' in param_name:
                            overall_trace3_bias += trace_p.item() / self.m_sampling
                        elif 'weight' in param_name:
                            overall_trace3_weight += trace_p.item() / self.m_sampling
                        else:
                            raise ValueError("The parameters are neither biases nor weights.")
        
        # evaluators[f'trace_fim_overall/{kind}_trace_bias'] = overall_trace1_bias + overall_trace2_bias + overall_trace3_bias
        evaluators[f'trace_fim_overall_{self.postfix}/{kind}_trace_weight'] = overall_trace1_weight + overall_trace2_weight + overall_trace3_weight
        # evaluators[f'trace_fim_overall/{kind}_trace'] = evaluators[f'trace_fim_overall/{kind}_trace_bias'] + evaluators[f'trace_fim_overall/{kind}_trace_weight']
        
        # evaluators[f'trace_fim_overall/{kind}_trace1_bias'] = overall_trace1_bias
        evaluators[f'trace_fim_overall_{self.postfix}/{kind}_trace1_weight'] = overall_trace1_weight
        # evaluators[f'trace_fim_overall/{kind}_trace1'] = overall_trace1_bias + overall_trace1_weight
        
        # evaluators[f'trace_fim_overall/{kind}_trace2_bias'] = overall_trace2_bias
        evaluators[f'trace_fim_overall_{self.postfix}/{kind}_trace2_weight'] = overall_trace2_weight
        # evaluators[f'trace_fim_overall/{kind}_trace2'] = overall_trace2_bias + overall_trace2_weight
        
        # evaluators[f'trace_fim_overall/{kind}_trace3_bias'] = overall_trace3_bias
        evaluators[f'trace_fim_overall_{self.postfix}/{kind}_trace3_weight'] = overall_trace3_weight
        # evaluators[f'trace_fim_overall/{kind}_trace3'] = overall_trace3_bias + overall_trace3_weight
        
        # evaluators[f'trace_fim_overall/{kind}_ratio_left_to_right_bias'] = overall_trace1_bias / (overall_trace2_bias + 1e-10)
        evaluators[f'trace_fim_overall_{self.postfix}/{kind}_ratio_left_to_right_weight'] = overall_trace1_weight / (overall_trace2_weight + 1e-10)
        # evaluators[f'trace_fim_overall/{kind}_ratio_left_to_right'] = (overall_trace1_bias + overall_trace1_weight) / (overall_trace2_bias + overall_trace2_weight + 1e-10)
        # evaluators[f'trace_fim_{kind}/overall_ratio_1_to_3'] = overall_trace1 / (overall_trace3 + 1e-10)
        # evaluators[f'trace_fim_{kind}/overall_ratio_2_to_3'] = overall_trace2 / (overall_trace3 + 1e-10)
        evaluators[f'steps/trace_fim_{self.postfix}'] = global_step
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


import os
from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np
class BaseAnalysis:
    def export(self, name):
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def clean_up(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a.to("cpu")
                del a
            except AttributeError:
                pass
        del self
        torch.cuda.empty_cache()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class RepresentationsSpectra(BaseAnalysis):
    # macierz grama czy macierz kowariancji?
    def __init__(self, model, loaders, modules_list, is_left_branch, layers=None, rpath='.', MAX_REPR_SIZE=8000):
        self.model = model
        self.loaders = loaders
        self.names_of_layers_to_analyze = layers if layers is not None else [n for n, _ in model.named_modules()]
        self.modules_list = modules_list
        self.handels = []
        self._insert_hooks()
        self.representations = {}
        self.MAX_REPR_SIZE = MAX_REPR_SIZE
        self.rpath = rpath
        # os.makedirs(self.rpath, exist_ok=True)
        self.attributes_on_gpu = ["model"]
        self.logger = None
        self.is_able = False
        self.device = next(model.parameters()).device
        self.is_left_branch = is_left_branch
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.subsampling = {}

    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            if self.is_able:
                output = output.flatten(start_dim=1)
                representation_size = output.shape[1]
                if name in self.subsampling:  # czy da się lepiej? Spytać Staszka. #TODO
                    output = torch.index_select(output, 1, self.subsampling[name].to(self.device))
                elif representation_size > self.MAX_REPR_SIZE:
                    self.subsampling[name] = torch.randperm(representation_size)[:self.MAX_REPR_SIZE].sort()[0]
                    output = torch.index_select(output, 1, self.subsampling[name].to(self.device))
                
                self.representations[name] = self.representations.get(name, []) + [output]
        return spectra_hook

    def _insert_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.names_of_layers_to_analyze:
                if any(isinstance(module, module_type) for module_type in self.modules_list):
                    self.handels.append(module.register_forward_hook(self._spectra_hook(name)))
                
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True

    @torch.no_grad()
    def collect_representations(self, kind, phase):
        self.model.eval()
        y_true = torch.empty((0,))
        with torch.no_grad():
            phase = f'{phase}_proper' if kind == 'proper' else f'{phase}_blurred'
            for data, y_data in self.loaders[phase]:
                x_true = data[0] if self.is_left_branch else data[1]
                x_true = x_true.to(self.device)
                _ = self.model(x_true)
                y_true = torch.cat((y_true, y_data))
        for name, rep in self.representations.items():
            self.representations[name] = torch.cat(rep, dim=0).detach()
        self.model.train()
        return y_true
    
    def collect_weights(self):
        named_weights = {n: p.reshape(p.size(0), -1) for n, p in self.model.named_parameters() if 'weight' in n and n in self.penalized_parameter_names}
        return named_weights

    def analysis(self, step, scope, phase, kind):
        main_prefix = f'ranks_representations_{"left" if self.is_left_branch else "right"}_branch_{phase}'
        postfix = f'____{scope}____{phase}'
        y_true = self.collect_representations(kind, phase)
        
        prefix = main_prefix
        evaluators1 = {}
        for name, rep in self.representations.items():  # internal representations
            name_dict = f'{prefix}/{name}{postfix}'
            rep = torch.cov(rep.T)
            # rep = rep.T @ rep
            rank = torch.linalg.matrix_rank(rep).item()
            evaluators1[name_dict] = rank
            
        self.plot(evaluators1, prefix, postfix)
        
        prefix = f'square_stable_{main_prefix}'
        evaluators2 = {}
        for name, rep in self.representations.items():  # internal representations
            name_dict = f'{prefix}/{name}{postfix}'
            rep = torch.cov(rep.T)
            singular_squared = torch.linalg.eig(rep)[0].float()
            square_stable_rank = singular_squared.sum() / max(singular_squared)
            evaluators2[name_dict] = square_stable_rank.item()
            
        self.plot(evaluators2, prefix, postfix)
        
        evaluators = evaluators1 | evaluators2
        
        variance_eucl(self.representations, y_true, evaluators, label="left" if self.is_left_branch else "right", phase=phase)  # da sie zastosować square stable rank tutaj?
        
        if phase == 'train':
            main_prefix = f'ranks_weights_{"left" if self.is_left_branch else "right"}_branch'
            postfix = f'____{scope}____{phase}'
            named_weights = self.collect_weights()
            
            prefix = main_prefix
            evaluators3 = {}
            for name, weights in named_weights.items():     # weights
                name_dict = f'{prefix}/{name}{postfix}'
                weights = torch.cov(weights)
                # weights = weights @ weights.T
                evaluators3[name_dict] = torch.linalg.matrix_rank(weights).item()
                
            self.plot(evaluators3, prefix, postfix)
            
            prefix = f'square_stable_{main_prefix}'
            evaluators4 = {}
            for name, weights in named_weights.items():     # weights
                name_dict = f'{prefix}/{name}{postfix}'
                weights = torch.cov(weights)
                singular_squared = torch.linalg.eig(weights)[0].float()
                square_stable_rank = singular_squared.sum() / max(singular_squared)
                evaluators4[name_dict] = square_stable_rank.item()
                
            self.plot(evaluators4, prefix, postfix)
            
            evaluators = evaluators | evaluators3 | evaluators4

        
        evaluators[f'steps/ranks_{"left" if self.is_left_branch else "right"}_{phase}'] = step
        self.logger.log_scalars(evaluators, step)
        self.representations = {}
        self.subsampling = {}
        torch.cuda.empty_cache()
        # self.clean_up()

    def plot(self, evaluators, prefix, postfix):
        plot_name = f'{prefix}_plots/{postfix}'
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(list(range(len(evaluators))), list(evaluators.values()), "o-")
        # print(list(evaluators.keys()))
        # Dodawanie tytułu i etykiet osi
        axs.set_title("Rank Across Layers")  # Dodaj tytuł wykresu
        axs.set_xlabel("Layer")  # Dodaj etykietę dla osi X
        axs.set_ylabel("Rank")  # Dodaj etykietę dla osi Y
        plot_images = {plot_name: fig}
        self.logger.log_plots(plot_images)
        # plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
        plt.close()
        
        
        
        
class DeadReLU:
    '''
    Gather dead activations
    '''
    def __init__(self, model, is_left_branch, is_able):
        self.model = model
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.modules_list = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.GELU]
        self.is_able = is_able
        self.nb_of_dead_relu = {}
        self.handels = []
        self.logger = None
        self._insert_hooks()
        self.is_left_branch = is_left_branch
        
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True
        
    def _deadrelu_hook(self, name):
        def deadrelu_hook(model, input, output):
            if self.is_able:
                output = output.flatten(start_dim=1)  # (N, D), D - liczba jednostek w reprezentacji, pojedyńcze skalary
                output = (output <= 0).sum(axis=0)  # (D, )
                if name not in self.nb_of_dead_relu:
                    self.nb_of_dead_relu[name] = output
                else:
                    self.nb_of_dead_relu[name] += output
        return deadrelu_hook

    def _insert_hooks(self):
        for name, module in self.model.named_modules():
            if any(isinstance(module, module_type) for module_type in self.modules_list):
                self.handels.append(module.register_forward_hook(self._deadrelu_hook(name)))
                    
    def at_the_epoch_end(self, phase, max_dataset, step):
        number = sum([(self.nb_of_dead_relu[name] == max_dataset).sum() for name in self.nb_of_dead_relu]) / sum([self.nb_of_dead_relu[name].shape[0] for name in self.nb_of_dead_relu])
        evaluators = {f'nb_of_dead_relu_units_{"left" if self.is_left_branch else "right"}_branch/overall_frac____epoch____{phase}': number}
        numbers = [(self.nb_of_dead_relu[name] == max_dataset).float().mean() for name in self.nb_of_dead_relu]
        for i, name in enumerate(self.nb_of_dead_relu):
            evaluators[f'nb_of_dead_relu_units_{"left" if self.is_left_branch else "right"}_branch/{name}_frac____epoch____{phase}'] = numbers[i]
        evaluators[f'steps/dead_relu_{"left" if self.is_left_branch else "right"}'] = step
        self.logger.log_scalars(evaluators, step)
        self.nb_of_dead_relu = {}
        torch.cuda.empty_cache()
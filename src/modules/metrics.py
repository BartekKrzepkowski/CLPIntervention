import math
from copy import deepcopy

import numpy as np
import torch

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES

def acc_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    acc = correct / y_pred.size(0)
    return acc


def prepare_evaluators(y_pred, y_true, loss):
    acc = acc_metric(y_pred, y_true)
    evaluators = {'loss': loss.item(), 'acc': acc}
    return evaluators


class RunStats(torch.nn.Module):
    def __init__(self, model, optim):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.model = model
        self.optim = optim
        self.model_trajectory_length_group = {k: 0.0 for k, _ in self.model.named_parameters() if _.requires_grad}
        self.model_trajectory_length_overall = 0.0
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        
    def forward(self, evaluators, distance_type):
        self.model.eval()
        self.count_dead_neurons(evaluators)
        self.model_trajectory_length(evaluators)
        self.distance_between_models(evaluators, distance_type)
        evaluators['run_stats/excessive_length_overall'] = evaluators['run_stats/model_trajectory_length_overall'] - evaluators[f'run_stats/distance_from initialization_{distance_type}']
        self.model.train()
        return evaluators
    
    def model_trajectory_length(self, evaluators, norm_type=2.0): # odłączyć liczenie normy gradientu od liczenia długości trajektorii
        '''
        Evaluates the model trajectory length.
        '''
        lr = self.optim.param_groups[-1]['lr']
        named_parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and n in self.allowed_parameter_names]
        grad_norm_per_layer = []
        weight_norm_per_layer = []
        for n, p in named_parameters:
            weight_norm_per = torch.norm(p.data, norm_type)
            evaluators[f'run_stats_model_weight_norm_squared/{n}'] = weight_norm_per.item() ** 2
            grad_norm_per = torch.norm(p.grad, norm_type) if p.grad is not None else torch.tensor(0.0)
            evaluators[f'run_stats_model_gradient_norm_squared/{n}'] = grad_norm_per.item() ** 2
            if n in self.allowed_parameter_names:
                weight_norm_per_layer.append(weight_norm_per)
                grad_norm_per_layer.append(grad_norm_per)
            self.model_trajectory_length_group[n] += lr * grad_norm_per.item()
            evaluators[f'run_stats_model_trajectory_length_group/{n}'] = self.model_trajectory_length_group[n]
            
        weight_norm = torch.norm(torch.stack(weight_norm_per_layer), norm_type).item()
        evaluators[f'run_stats/model_weight_norm_squared_overall'] = weight_norm ** 2
        grad_norm = torch.norm(torch.stack(grad_norm_per_layer), norm_type).item()
        evaluators[f'run_stats/model_gradient_norm_squared_overall'] = grad_norm ** 2
        self.model_trajectory_length_overall += lr * grad_norm
        evaluators['run_stats/model_trajectory_length_overall'] = self.model_trajectory_length_overall
    
    def distance_between_models(self, evaluators, distance_type):
        def distance_between_models_l2(named_parameters1, named_parameters2, norm_type=2.0):
            """
            Returns the l2 distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                dist = torch.norm(p1-p2, norm_type)
                if n1 in self.allowed_parameter_names:
                    distances.append(dist)
                evaluators[f'run_stats_distance_from initialization_l2/{n1}'] = dist.item()
            distance = torch.norm(torch.stack(distances), norm_type)
            evaluators['run_stats/distance_from initialization_l2'] = distance.item()
        
        def distance_between_models_cosine(named_parameters1, named_parameters2):
            """
            Returns the cosine distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                1 / 0
                distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
            return distance.item()

        """
        Returns the distance between two models.
        """
        named_parameters1 = [(n, p) for n, p in self.model_zero.named_parameters() if p.requires_grad]
        named_parameters2 = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2)
        elif distance_type == 'cosine':
            distance_between_models_cosine(named_parameters1, named_parameters2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        
        
    def count_dead_neurons(self, evaluators):
        dead_neurons_overall = 0
        all_neurons = 0

        # Iterate over the model's modules (layers)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    dead_neurons = torch.sum(torch.all(module.weight.abs() < 1e-14, dim=1)).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    dead_neurons_overall += dead_neurons
                    all_neurons += neurons
                    
            if isinstance(module, torch.nn.Conv2d):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    cond = torch.all(module.weight.abs() < 1e-14, dim=1).all(dim=1).all(dim=1)
                    dead_neurons = torch.sum(cond).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    dead_neurons_overall += dead_neurons
                    all_neurons += neurons

        evaluators['run_stats/dead_neurons_overall'] = dead_neurons_overall / all_neurons
        
        
class RunStatsBiModal(torch.nn.Module):
    def __init__(self, model, optim):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.last_model = deepcopy(model)
        self.model = model
        self.optim = optim
        self.left_branch_trajectory_length_group = {n: 0.0 for n, p in self.model.named_parameters() if p.requires_grad and 'left_branch' in n}
        self.right_branch_trajectory_length_group = {n: 0.0 for n, p in self.model.named_parameters() if p.requires_grad and 'right_branch' in n}
        self.main_branch_trajectory_length_group = {n: 0.0 for n, p in self.model.named_parameters() if p.requires_grad and 'main_branch' in n}
        self.model_trajectory_length_overall = 0.0
        self.left_branch_trajectory_length_overall = 0.0
        self.right_branch_trajectory_length_overall = 0.0
        self.main_branch_trajectory_length_overall = 0.0
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.logger = None
        self.eps = torch.tensor(1e-9)
        
    def forward(self, distance_type, global_step):          
        evaluators = defaultdict(float)
        self.model.eval()
        # self.count_dead_neurons(evaluators)
        self.model_trajectory_length(evaluators)
        self.distance_between_models(self.model, self.model_zero, evaluators, distance_type, dist_label='distance_from initialization')
        self.distance_between_models(self.model, self.last_model, evaluators, distance_type, dist_label='distance_from_last_checkpoint')
        self.distance_between_branches(self.model, self.model, evaluators, distance_type, dist_label='distance_between_branches')
        self.distance_between_branches(self.model, self.model, evaluators, distance_type='angle', dist_label='distance_between_branches')
        self.last_model = deepcopy(self.model)
        evaluators['steps/run_stats'] = global_step
        self.model.train()
        self.logger.log_scalars(evaluators, global_step)   
    
    def model_trajectory_length(self, evaluators, norm_type=2.0): # odłączyć liczenie normy gradientu od liczenia długości trajektorii
        '''
        Evaluates the model trajectory length.
        '''
        lr = self.optim.param_groups[-1]['lr']
        
        named_parameters1 = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and 'left_branch' in n]
        weight_norm_layers = []
        grad_norm_layers = []
        weight_sign_proportion_mean1_bias = []
        weight_sign_proportion_std1_bias = []
        weight_sign_proportion_mean1_weight = []
        weight_sign_proportion_std1_weight = []
        neurons_counter1_bias = []
        neurons_counter1_weight = []
        for n, p in named_parameters1:
            weight_norm_per_layer = torch.norm(p.data, norm_type)
            evaluators[f'run_stats_model_weight_norm_squared/left_branch_{n}'] = weight_norm_per_layer.item() ** 2
            grad_norm_per_layer = torch.norm(p.grad, norm_type) if p.grad is not None else torch.tensor(0.0)
            evaluators[f'run_stats_model_gradient_norm_squared/left_branch_{n}'] = grad_norm_per_layer.item() ** 2
            evaluators[f'run_stats_model_grad_weight_norm_ratio_squared/left_branch_{n}'] = evaluators[f'run_stats_model_gradient_norm_squared/left_branch_{n}'] / (1e-9 + evaluators[f'run_stats_model_weight_norm_squared/left_branch_{n}'])
            mean, std = self.weights_sign_proportion(p.data)
            evaluators[f'run_stats_model_weights_sign_proporion_mean/left_branch_{n}'] = mean
            evaluators[f'run_stats_model_weights_sign_proporion_std/left_branch_{n}'] = std
            if n in self.allowed_parameter_names:
                weight_norm_layers.append(weight_norm_per_layer)
                grad_norm_layers.append(grad_norm_per_layer)
                if 'bias' in n:
                    weight_sign_proportion_mean1_bias.append(mean)
                    weight_sign_proportion_std1_bias.append(std)
                    neurons_counter1_bias.append(p.data.shape[0])
                elif 'weight' in n:
                    weight_sign_proportion_mean1_weight.append(mean)
                    weight_sign_proportion_std1_weight.append(std)
                    neurons_counter1_weight.append(p.data.shape[0])
                else:
                    raise ValueError("The parameters are neither biases nor weights.")
                
            self.left_branch_trajectory_length_group[n] += lr * grad_norm_per_layer.item()
            evaluators[f'run_stats_model_trajectory_length_group/left_branch_{n}'] = self.left_branch_trajectory_length_group[n]
            
        weight_norm1 = torch.norm(torch.stack(weight_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/left_branch_weight_norm_squared'] = weight_norm1 ** 2
        grad_norm1 = torch.norm(torch.stack(grad_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/left_branch_gradient_norm_squared'] = grad_norm1 ** 2
        evaluators[f'run_stats_overall/left_branch_grad_weight_norm_ratio_squared'] = evaluators[f'run_stats_overall/left_branch_gradient_norm_squared'] / (1e-9 + evaluators[f'run_stats_overall/left_branch_weight_norm_squared'])
        self.left_branch_trajectory_length_overall += lr * grad_norm1
        evaluators['run_stats_overall/left_branch_trajectory_length'] = self.left_branch_trajectory_length_overall
        # sign proportion
        overall_mean_bias, overall_std_bias = self.overall_mean_and_std(weight_sign_proportion_mean1_bias, weight_sign_proportion_std1_bias, neurons_counter1_bias)
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_mean_bias'] = overall_mean_bias
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_std_bias'] = overall_std_bias
        overall_mean_weight, overall_std_weight = self.overall_mean_and_std(weight_sign_proportion_mean1_weight, weight_sign_proportion_std1_weight, neurons_counter1_weight)
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_mean_weight'] = overall_mean_weight
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_std_weight'] = overall_std_weight
        weight_sign_proportion_mean1 = weight_sign_proportion_mean1_bias + weight_sign_proportion_mean1_weight
        weight_sign_proportion_std1 = weight_sign_proportion_std1_bias + weight_sign_proportion_std1_weight
        neurons_counter1 = neurons_counter1_bias + neurons_counter1_weight
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean1, weight_sign_proportion_std1, neurons_counter1)
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_mean'] = overall_mean
        evaluators['run_stats_overall/left_branch_weights_sign_proporion_std'] = overall_std
        #-----------------------------------------------------------------------------------------------------------------
        named_parameters2 = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and 'right_branch' in n]
        weight_norm_layers = []
        grad_norm_layers = []
        weight_sign_proportion_mean2_bias = []
        weight_sign_proportion_std2_bias = []
        weight_sign_proportion_mean2_weight = []
        weight_sign_proportion_std2_weight = []
        neurons_counter2_bias = []
        neurons_counter2_weight = []
        for n, p in named_parameters2:
            weight_norm_per_layer = torch.norm(p.data, norm_type)
            evaluators[f'run_stats_model_weight_norm_squared/right_branch_{n}'] = weight_norm_per_layer.item() ** 2
            grad_norm_per_layer = torch.norm(p.grad, norm_type) if p.grad is not None else torch.tensor(0.0)
            evaluators[f'run_stats_model_gradient_norm_squared/right_branch_{n}'] = grad_norm_per_layer.item() ** 2
            mean, std = self.weights_sign_proportion(p.data)
            evaluators[f'run_stats_model_weights_sign_proporion_mean/right_branch_{n}'] = mean
            evaluators[f'run_stats_model_weights_sign_proporion_std/right_branch_{n}'] = std
            # evaluators[f'run_stats_model_grad_weight_norm_ratio_squared/right_branch_{n}'] = evaluators[f'run_stats_model_gradient_norm_squared/right_branch_{n}'] / (1e-9 + evaluators[f'run_stats_model_weight_norm_squared/right_branch_{n}'])
            if n in self.allowed_parameter_names:
                weight_norm_layers.append(weight_norm_per_layer)
                grad_norm_layers.append(grad_norm_per_layer)
                if 'bias' in n:
                    weight_sign_proportion_mean2_bias.append(mean)
                    weight_sign_proportion_std2_bias.append(std)
                    neurons_counter2_bias.append(p.data.shape[0])
                elif 'weight' in n:
                    weight_sign_proportion_mean2_weight.append(mean)
                    weight_sign_proportion_std2_weight.append(std)
                    neurons_counter2_weight.append(p.data.shape[0])
                else:
                    raise ValueError("The parameters are neither biases nor weights.")
            self.right_branch_trajectory_length_group[n] += lr * grad_norm_per_layer.item()
            evaluators[f'run_stats_model_trajectory_length_group/right_branch_{n}'] = self.right_branch_trajectory_length_group[n]
            
        weight_norm2 = torch.norm(torch.stack(weight_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/right_branch_weight_norm_squared'] = weight_norm2 ** 2
        grad_norm2 = torch.norm(torch.stack(grad_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/right_branch_gradient_norm_squared'] = grad_norm2 ** 2
        # evaluators[f'run_stats/right_branch_grad_weight_norm_ratio_squared_overall'] = evaluators[f'run_stats/right_branch_gradient_norm_squared_overall'] / (1e-9 + evaluators[f'run_stats/right_branch_weight_norm_squared_overall'])
        self.right_branch_trajectory_length_overall += lr * grad_norm2
        evaluators['run_stats_overall/right_branch_trajectory_length'] = self.right_branch_trajectory_length_overall
        # sign proportion
        overall_mean_bias, overall_std_bias = self.overall_mean_and_std(weight_sign_proportion_mean2_bias, weight_sign_proportion_std2_bias, neurons_counter2_bias)
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_mean_bias'] = overall_mean_bias
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_std_bias'] = overall_std_bias
        overall_mean_weight, overall_std_weight = self.overall_mean_and_std(weight_sign_proportion_mean2_weight, weight_sign_proportion_std2_weight, neurons_counter2_weight)
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_mean_weight'] = overall_mean_weight
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_std_weight'] = overall_std_weight
        weight_sign_proportion_mean2 = weight_sign_proportion_mean2_bias + weight_sign_proportion_mean2_weight
        weight_sign_proportion_std2 = weight_sign_proportion_std2_bias + weight_sign_proportion_std2_weight
        neurons_counter2 = neurons_counter2_bias + neurons_counter2_weight
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean2, weight_sign_proportion_std2, neurons_counter2)
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_mean'] = overall_mean
        evaluators['run_stats_overall/right_branch_weights_sign_proporion_std'] = overall_std
        #-----------------------------------------------------------------------------------------------------------------
        named_parameters3 = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and 'main_branch' in n]
        weight_norm_layers = []
        grad_norm_layers = []
        weight_sign_proportion_mean3_bias = []
        weight_sign_proportion_std3_bias = []
        weight_sign_proportion_mean3_weight = []
        weight_sign_proportion_std3_weight = []
        neurons_counter3_bias = []
        neurons_counter3_weight = []
        for n, p in named_parameters3:
            weight_norm_per_layer = torch.norm(p.data, norm_type)
            evaluators[f'run_stats_model_weight_norm_squared/main_branch_{n}'] = weight_norm_per_layer.item() ** 2
            grad_norm_per_layer = torch.norm(p.grad, norm_type) if p.grad is not None else torch.tensor(0.0)
            evaluators[f'run_stats_model_gradient_norm_squared/main_branch_{n}'] = grad_norm_per_layer.item() ** 2
            mean, std = self.weights_sign_proportion(p.data)
            evaluators[f'run_stats_model_weights_sign_proporion_mean/main_branch_{n}'] = mean
            evaluators[f'run_stats_model_weights_sign_proporion_std/main_branch_{n}'] = std
            # evaluators[f'run_stats_model_grad_weight_norm_ratio_squared/main_branch_{n}'] = evaluators[f'run_stats_model_gradient_norm_squared/main_branch_{n}'] / (1e-9 + evaluators[f'run_stats_model_weight_norm_squared/main_branch_{n}'])
            if n in self.allowed_parameter_names:
                weight_norm_layers.append(weight_norm_per_layer)
                grad_norm_layers.append(grad_norm_per_layer)
                if 'bias' in n:
                    weight_sign_proportion_mean3_bias.append(mean)
                    weight_sign_proportion_std3_bias.append(std)
                    neurons_counter3_bias.append(p.data.shape[0])
                elif 'weight' in n:
                    weight_sign_proportion_mean3_weight.append(mean)
                    weight_sign_proportion_std3_weight.append(std)
                    neurons_counter3_weight.append(p.data.shape[0])
                else:
                    raise ValueError("The parameters are neither biases nor weights.")
            self.main_branch_trajectory_length_group[n] += lr * grad_norm_per_layer.item()
            evaluators[f'run_stats_model_trajectory_length_group/main_branch_{n}'] = self.main_branch_trajectory_length_group[n]
            
        weight_norm3 = torch.norm(torch.stack(weight_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/main_branch_weight_norm_squared'] = weight_norm3 ** 2
        grad_norm3 = torch.norm(torch.stack(grad_norm_layers), norm_type).item()
        evaluators[f'run_stats_overall/main_branch_gradient_norm_squared'] = grad_norm3 ** 2
        # evaluators[f'run_stats/main_branch_grad_weight_norm_ratio_squared_overall'] = evaluators[f'run_stats/main_branch_gradient_norm_squared_overall'] / (1e-9 + evaluators[f'run_stats/main_branch_weight_norm_squared_overall'])
        self.main_branch_trajectory_length_overall += lr * grad_norm3
        evaluators['run_stats_overall/main_branch_trajectory_length'] = self.main_branch_trajectory_length_overall
        # sign proportion
        overall_mean_bias, overall_std_bias = self.overall_mean_and_std(weight_sign_proportion_mean3_bias, weight_sign_proportion_std3_bias, neurons_counter3_bias)
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_mean_bias'] = overall_mean_bias
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_std_bias'] = overall_std_bias
        overall_mean_weight, overall_std_weight = self.overall_mean_and_std(weight_sign_proportion_mean3_weight, weight_sign_proportion_std3_weight, neurons_counter3_weight)
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_mean_weight'] = overall_mean_weight
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_std_weight'] = overall_std_weight
        weight_sign_proportion_mean3 = weight_sign_proportion_mean3_bias + weight_sign_proportion_mean3_weight
        weight_sign_proportion_std3 = weight_sign_proportion_std3_bias + weight_sign_proportion_std3_weight
        neurons_counter3 = neurons_counter3_bias + neurons_counter3_weight
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean3, weight_sign_proportion_std3, neurons_counter3)
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_mean'] = overall_mean
        evaluators['run_stats_overall/main_branch_weights_sign_proporion_std'] = overall_std
        #-----------------------------------------------------------------------------------------------------------------
        evaluators[f'run_stats_overall/left_to_right_gradient_norm_squared'] = grad_norm1 ** 2 / (grad_norm2 ** 2 + 1e-9)
        evaluators[f'run_stats_overall/left_to_main_gradient_norm_squared'] = grad_norm1 ** 2 / (grad_norm3 ** 2 + 1e-9)
        evaluators[f'run_stats_overall/right_to_main_gradient_norm_squared'] = grad_norm2 ** 2 / (grad_norm3 ** 2 + 1e-9)
        weight_norm = torch.norm(torch.stack([torch.tensor(weight_norm1), torch.tensor(weight_norm2), torch.tensor(weight_norm3)]), norm_type).item()
        evaluators[f'run_stats_overall/model_weight_norm_squared'] = weight_norm ** 2
        grad_norm = torch.norm(torch.stack([torch.tensor(grad_norm1), torch.tensor(grad_norm2), torch.tensor(grad_norm3)]), norm_type).item()
        evaluators[f'run_stats_overall/model_gradient_norm_squared'] = grad_norm ** 2
        # evaluators[f'run_stats/model_grad_weight_norm_ratio_squared_overall'] = evaluators[f'run_stats/model_gradient_norm_squared_overall'] / (1e-9 + evaluators[f'run_stats/model_weight_norm_squared_overall'])
        self.model_trajectory_length_overall += lr * grad_norm
        evaluators['run_stats_overall/model_trajectory_length'] = self.model_trajectory_length_overall
        # sign proportion
        neurons_counter_bias = neurons_counter1_bias + neurons_counter2_bias + neurons_counter3_bias
        weight_sign_proportion_mean_bias = weight_sign_proportion_mean1_bias + weight_sign_proportion_mean2_bias + weight_sign_proportion_mean3_bias
        weight_sign_proportion_std_bias = weight_sign_proportion_std1_bias + weight_sign_proportion_std2_bias + weight_sign_proportion_std3_bias
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean_bias, weight_sign_proportion_std_bias, neurons_counter_bias)
        evaluators['run_stats_overall/model_weights_sign_proporion_mean_bias'] = overall_mean
        evaluators['run_stats_overall/model_weights_sign_proporion_std_bias'] = overall_std
        neurons_counter_weight = neurons_counter1_weight + neurons_counter2_weight + neurons_counter3_weight
        weight_sign_proportion_mean_weight = weight_sign_proportion_mean1_weight + weight_sign_proportion_mean2_weight + weight_sign_proportion_mean3_weight
        weight_sign_proportion_std_weight = weight_sign_proportion_std1_weight + weight_sign_proportion_std2_weight + weight_sign_proportion_std3_weight
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean_weight, weight_sign_proportion_std_weight, neurons_counter_weight)
        evaluators['run_stats_overall/model_weights_sign_proporion_mean_weight'] = overall_mean
        evaluators['run_stats_overall/model_weights_sign_proporion_std_weight'] = overall_std
        neurons_counter = neurons_counter_bias + neurons_counter_weight
        weight_sign_proportion_mean = weight_sign_proportion_mean_bias + weight_sign_proportion_mean_weight
        weight_sign_proportion_std = weight_sign_proportion_std_bias + weight_sign_proportion_std_weight
        overall_mean, overall_std = self.overall_mean_and_std(weight_sign_proportion_mean, weight_sign_proportion_std, neurons_counter)
        evaluators['run_stats_overall/model_weights_sign_proporion_mean'] = overall_mean
        evaluators['run_stats_overall/model_weights_sign_proporion_std'] = overall_std
        
        
    
    def distance_between_models(self, model1, model2, evaluators, distance_type, dist_label):
        def distance_between_models_l2(named_parameters1, named_parameters2, dist_label, norm_type=2.0, branch_name=None):
            """
            Returns the l2 distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                dist = torch.norm(p1-p2, norm_type)
                if n1 in self.allowed_parameter_names:
                    distances.append(dist)
                evaluators[f'run_stats_{dist_label}_l2/{branch_name}_{n1}'] = dist.item()
            distance = torch.norm(torch.stack(distances), norm_type)
            evaluators[f'run_stats_overall/{branch_name}_{dist_label}_l2'] = distance.item()
        
        def distance_between_models_cosine(named_parameters1, named_parameters2, dist_label, branch_name):
            """
            TODO
            Returns the cosine distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
            return distance.item()

        """
        Returns the distance between two models.
        """
        named_parameters1 = [(n, p) for n, p in model1.named_parameters() if p.requires_grad and 'left_branch' in n]
        named_parameters2 = [(n, p) for n, p in model2.named_parameters() if p.requires_grad and 'left_branch' in n]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2, dist_label=dist_label, branch_name='left_branch')
        elif distance_type == 'cosine':
            pass
            # distance_between_models_cosine(named_parameters1, named_parameters2, net_nb=1)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        #-----------------------------------------------------------------------------------------------------------------
        named_parameters1 = [(n, p) for n, p in model1.named_parameters() if p.requires_grad and 'right_branch' in n]
        named_parameters2 = [(n, p) for n, p in model2.named_parameters() if p.requires_grad and 'right_branch' in n]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2, dist_label=dist_label, branch_name='right_branch')
        elif distance_type == 'cosine':
            pass
            # distance_between_models_cosine(named_parameters1, named_parameters2, net_nb=2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        #-----------------------------------------------------------------------------------------------------------------
        named_parameters1 = [(n, p) for n, p in model1.named_parameters() if p.requires_grad and 'main_branch' in n]
        named_parameters2 = [(n, p) for n, p in model2.named_parameters() if p.requires_grad and 'main_branch' in n]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2, dist_label=dist_label, branch_name='main_branch')
        elif distance_type == 'cosine':
            pass
            # distance_between_models_cosine(named_parameters1, named_parameters2, net_nb=3)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        #-----------------------------------------------------------------------------------------------------------------
        evaluators[f'run_stats_overall/model_{dist_label}_l2'] = np.sqrt(sum([evaluators[f'run_stats_overall/{branch_name}_{dist_label}_l2'] ** 2 for branch_name in ['left_branch', 'right_branch', 'main_branch']]))
        
    def distance_between_branches(self, model1, model2, evaluators, distance_type, dist_label):
        def distance_between_models_l2(named_parameters1, named_parameters2, dist_label, branch_name=None, norm_type=2.0):
            """
            Returns the l2 distance between two models.
            """
            distances = []
            for (n1, p1), (n2, p2) in zip(named_parameters1, named_parameters2):
                # print(f"The same names? {dist_label}:", '.'.join(n1.split('.')[1:]) == '.'.join(n2.split('.')[1:]))
                dist = torch.norm(p1-p2, norm_type)
                if n1 in self.allowed_parameter_names:
                    distances.append(dist)
                n1 = '.'.join(n1.split('.')[1:])
                evaluators[f'run_stats_{dist_label}_l2/{branch_name}_{n1}'] = dist.item()
            distance = torch.norm(torch.stack(distances), norm_type)
            evaluators[f'run_stats_overall/{dist_label}_l2'] = distance.item()
            
        def distance_between_models_angle(named_parameters1, named_parameters2, named_parameters3, named_parameters4, dist_label, branch_name, norm_type=2.0):
            """
            TODO
            Returns the cosine distance between two models.
            """
            left_norms = []
            right_norms = []
            global_numerator = 0.0
            for (n1, p1), (_, p2), (_, p3), (_, p4) in zip(named_parameters1, named_parameters2, named_parameters3, named_parameters4):
                v1 = (p1 - p3).view(-1)
                v2 = (p2 - p4).view(-1)
                numerator = (v1 @ v2)
                left_norm, right_norm = v1.norm(norm_type), v2.norm(norm_type)
                if n1 in self.allowed_parameter_names:
                    global_numerator += numerator
                    left_norms.append(left_norm)
                    right_norms.append(right_norm)
                n1 = '.'.join(n1.split('.')[1:])
                evaluators[f'run_stats_{dist_label}_angle_in_radians_over_pi/{branch_name}_{n1}'] = torch.arccos(numerator / (left_norm * right_norm + self.eps)).item() / torch.pi
            overall_left_norm = torch.norm(torch.stack(left_norms), norm_type)
            overall_right_norm = torch.norm(torch.stack(right_norms), norm_type)
            evaluators[f'run_stats_overall/{dist_label}_angle_in_radians_over_pi'] = torch.arccos(global_numerator / (overall_left_norm * overall_right_norm + self.eps)).item() / torch.pi
            
        """
        Returns the distance between two branches.
        """
        named_parameters1 = [(n, p) for n, p in model1.named_parameters() if p.requires_grad and 'left_branch' in n]
        named_parameters2 = [(n, p) for n, p in model2.named_parameters() if p.requires_grad and 'right_branch' in n]
        named_parameters3 = [(n, p) for n, p in self.model_zero.named_parameters() if p.requires_grad and 'left_branch' in n]
        named_parameters4 = [(n, p) for n, p in self.model_zero.named_parameters() if p.requires_grad and 'right_branch' in n]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2, dist_label=dist_label, branch_name='both_branches')
        elif distance_type == 'angle':
            distance_between_models_angle(named_parameters1, named_parameters2, named_parameters3, named_parameters4, dist_label=dist_label, branch_name='both_branches')
            # distance_between_models_cosine(named_parameters1, named_parameters2, net_nb=2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        
        
    def weights_sign_proportion(self, weights):
        weights = (weights.cpu() >= 0).reshape(weights.shape[0], -1)  # (liczba neuronów, liczba parametrów)
        distrib = (2 * weights.sum(axis=1) - weights.shape[1]) / weights.shape[1]  # (b-a)/(a+b)
        mean = distrib.mean().item()
        std = distrib.std(correction=0).item()
        return mean, std
    
    
    def overall_mean_and_std(self, means, stds, counts):
        """
        Oblicza odchylenie standardowe dla sumy grup danych.

        :param groups: Lista krotek, gdzie każda krotka zawiera średnią (mean) i odchylenie standardowe (std) danej grupy.
        :return: Odchylenie standardowe sumy grup.
        """
        assert len(means) == len(stds) and len(means) == len(counts) and len(stds) == len(counts), "Lengths of lists are not equal."
        if len(counts) == 0:
            return 0.0, 0.0
        # Wartości początkowe dla sumy średnich i sumy liczebności
        total_mean_sum = 0
        total_count = 0

        # Obliczanie sumy średnich i sumy liczebności
        for mean, std, count in zip(means, stds, counts):
            total_mean_sum += mean * count
            total_count += count

        # Średnia dla całkowitej sumy
        overall_mean = total_mean_sum / total_count
        # Wartość początkowa dla sumy wariancji
        variance_sum = 0

        # Obliczanie sumy wariancji
        for mean, std, count in zip(means, stds, counts):
            variance_sum += ((std ** 2) * count) + count * ((mean - overall_mean) ** 2)

        # Wariancja dla sumy grup
        overall_variance = variance_sum / total_count

        # Odchylenie standardowe dla sumy grup
        return overall_mean, np.sqrt(overall_variance)
        
        
        
    def count_dead_neurons(self, evaluators):
        net1_dead_neurons_overall = 0
        net1_all_neurons = 0
        net2_dead_neurons_overall = 0
        net2_all_neurons = 0
        net3_dead_neurons_overall = 0
        net3_all_neurons = 0

        # Iterate over the model's modules (layers)
        for name, module in self.model.net1.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    dead_neurons = torch.sum(torch.all(module.weight.abs() < 1e-10, dim=1)).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/net1_{name}'] = dead_neurons / neurons
                    net1_dead_neurons_overall += dead_neurons
                    net1_all_neurons += neurons
                    
            if isinstance(module, torch.nn.Conv2d):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    cond = torch.all(module.weight.abs() < 1e-10, dim=1).all(dim=1).all(dim=1)
                    dead_neurons = torch.sum(cond).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/net1_{name}'] = dead_neurons / neurons
                    net1_dead_neurons_overall += dead_neurons
                    net1_all_neurons += neurons
        evaluators['run_stats/net1_dead_neurons_overall'] = net1_dead_neurons_overall / net1_all_neurons
        #-----------------------------------------------------------------------------------------------------------------        
        # Iterate over the model's modules (layers)
        for name, module in self.model.net1.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    dead_neurons = torch.sum(torch.all(module.weight.abs() < 1e-14, dim=1)).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/net2_{name}'] = dead_neurons / neurons
                    net2_dead_neurons_overall += dead_neurons
                    net2_all_neurons += neurons
                    
            if isinstance(module, torch.nn.Conv2d):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    cond = torch.all(module.weight.abs() < 1e-14, dim=1).all(dim=1).all(dim=1)
                    dead_neurons = torch.sum(cond).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/net2_{name}'] = dead_neurons / neurons
                    net2_dead_neurons_overall += dead_neurons
                    net2_all_neurons += neurons
        evaluators['run_stats/net2_dead_neurons_overall'] = net2_dead_neurons_overall / net2_all_neurons
        #-----------------------------------------------------------------------------------------------------------------        
        # Iterate over the model's modules (layers)
        for name, module in self.model.net1.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    dead_neurons = torch.sum(torch.all(module.weight.abs() < 1e-14, dim=1)).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    net3_dead_neurons_overall += dead_neurons
                    net3_all_neurons += neurons
                    
            if isinstance(module, torch.nn.Conv2d):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    cond = torch.all(module.weight.abs() < 1e-14, dim=1).all(dim=1).all(dim=1)
                    dead_neurons = torch.sum(cond).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    net3_dead_neurons_overall += dead_neurons
                    net3_all_neurons += neurons
        evaluators['run_stats/net3_dead_neurons_overall'] = net3_dead_neurons_overall / net3_all_neurons
        #-----------------------------------------------------------------------------------------------------------------
        evaluators['run_stats/dead_neurons_overall'] = (net1_dead_neurons_overall + net2_dead_neurons_overall + net3_dead_neurons_overall) / (net1_all_neurons + net2_all_neurons + net3_all_neurons)


class CosineAlignments:
    def __init__(self, model, loader, criterion) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = next(model.parameters()).device

    def calc_variance(self, n):
        gs = torch.tensor(self.gather_gradients(n))
        gdv = 0.
        for i in range(n):
            for j in range(i+1, n):
                gdv += 1 - torch.dot(gs[i], gs[j]) / torch.norm(gs[i], gs[j])
        gdv /= 2 / (n * (n - 1))
        return gdv


    def gather_gradients(self, n, device):
        gs = []
        for i, (x_true, y_true) in enumerate(self.loader):
            if i >= n: break
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            self.criterion(y_pred, y_true).backward()
            g = [p.grad for p in self.model.parameters() if p.requires_grad]
            gs.append(g)
            self.model.zero_grad()
        return gs
    
def max_eigenvalue(model, loss_fn, data, target):
    # Set model to evaluation mode
    model.eval()
    # Create a variable from the data
    data = torch.autograd.Variable(data, requires_grad=True)
    # Compute the loss
    loss = loss_fn(model(data), target)
    # Compute the gradients
    grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            retain_graph=True,
            create_graph=True)
    # Get the gradients of the weights
    grads = torch.cat([g.reshape(-1) for g in grads])
    # Create a vector of ones with the same size as the gradients
    v = torch.ones(grads.size()).to(grads.device)
    # Compute the Hessian-vector product
    Hv = torch.autograd.grad(grads, model.parameters(), grad_outputs=v, retain_graph=True)
    # Concatenate the Hessian-vector product into a single vector
    Hv = torch.cat([h.reshape(-1) for h in Hv])
    # Compute the maximum eigenvalue using the power iteration method
    for _ in range(100):
        v = Hv / torch.norm(Hv)
        Hv = torch.autograd.grad(grads, model.parameters(), grad_outputs=v, retain_graph=True)
        Hv = torch.cat([h.reshape(-1) for h in Hv])

    return (v * Hv).sum()
        

import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering 
    
class PerSampleGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss, predictions

    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        self.model.eval()
        num_classes = y_true1.max() + 1
        matrices = {
            'similarity_11': {},
            'graham_11': {},
            'cov_11': {},
        }
        scalars = {
            'trace_of_cov_11': {},
            # 'rank_of_gradients_11': {},
            # 'rank_of_similarity_11': {},
            # # 'rank_of_weights': {},
            # 'cumm_gradients_rank_11': {},
            # 'gradients_subspace_dim_11': {},
        }
        prediction_stats = {}
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names}
        # buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        # print(list(buffers.keys()))
        buffers = {}
        
        # scalars['cumm_max_rank_of_weights'] = {k: min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights'] = {k: self.matrix_rank(v) / min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights']['concatenated_weights'] = sum(scalars['rank_of_weights'][tag] * scalars['cumm_max_rank_of_weights'][tag] for tag in scalars['rank_of_weights']) / sum(scalars[f'cumm_max_rank_of_weights'].values())
        # del scalars['cumm_max_rank_of_weights']
            
        ft_per_sample_grads1, y_pred1 = self.ft_criterion(params, buffers, x_true1, y_true1)
        y_pred_label1 = torch.argmax(y_pred1.data.squeeze(), dim=1)
        ft_per_sample_grads1 = {k1: v.detach().data for k1, v in ft_per_sample_grads1.items()}
        concatenated_weights1 = torch.empty((x_true1.shape[0], 0), device=x_true1.device)
        if x_true2 is not None:
            matrices.update({
                    'similarity_22': {},
                    'graham_22': {},
                    'cov_22': {},
                    'similarity_12': {},
                    'graham_12': {},
                    'cov_12': {},
                })
            scalars.update({
                    'trace_of_cov_22': {},
                    # 'rank_of_gradients_22': {},
                    # 'rank_of_similarity_22': {},
                    # 'rank_of_similarity_12': {},
                    # 'cumm_gradients_rank_22': {},
                    # 'gradients_subspace_dim_22': {},
                })
            ft_per_sample_grads2, y_pred2 = self.ft_criterion(params, buffers, x_true2, y_true2)
            y_pred_label2 = torch.argmax(y_pred2.data.squeeze(), dim=1)
            ft_per_sample_grads2 = {k2: v.detach().data for k2, v in ft_per_sample_grads2.items()}
            concatenated_weights2 = torch.empty((x_true2.shape[0], 0), device=x_true2.device)
        
        for idx in range(num_classes):
            idxs_mask1 = y_true1 == idx
            prediction_stats[f'misclassification_1_{idx}'] = (y_pred_label1[idxs_mask1] != y_true1[idxs_mask1]).float().mean().item()
            if x_true2 is not None:
                y_prob1 = torch.nn.functional.softmax(y_pred1.data.squeeze(), dim=1)
                y_prob2 = torch.nn.functional.softmax(y_pred2.data.squeeze(), dim=1)
                prediction_stats[f'misclassification_2_{idx}'] = (y_pred_label2[idxs_mask1] != y_true2[idxs_mask1]).float().mean().item()
                prediction_stats[f'mean_prob_discrepancy_{idx}'] = (y_prob1[idxs_mask1][:, idx]  - y_prob2[idxs_mask1][:, idx]).float().mean().item() 
        
        for k in ft_per_sample_grads1:
            normed_ft_per_sample_grad1, concatenated_weights1 = self.prepare_variables(ft_per_sample_grads1, concatenated_weights1, scalars, tag=k, ind='11')
            if x_true2 is not None:
                normed_ft_per_sample_grad2, concatenated_weights2 = self.prepare_variables(ft_per_sample_grads2, concatenated_weights2, scalars, tag=k, ind='22')
                self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_ft_per_sample_grad2, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=True, ind='22')
                self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=False, ind='12')
            
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_ft_per_sample_grad1, normed_ft_per_sample_grad1, matrices, scalars, tag=k, hermitian=True, ind='11')
            
        normed_concatenated_weights1 = self.prepare_concatenated_weights(ft_per_sample_grads1, concatenated_weights1, scalars, ind='11')
        if x_true2 is not None:
            normed_concatenated_weights2 = self.prepare_concatenated_weights(ft_per_sample_grads2, concatenated_weights2, scalars, ind='22')
            self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_concatenated_weights2, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='22')
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_concatenated_weights1, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=False, ind='12')
        
        self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_concatenated_weights1, normed_concatenated_weights1, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='11')
        # del scalars['cumm_gradients_rank_11']
        # if x_true2 is not None:
        #     del scalars['cumm_gradients_rank_22']
        self.model.train()
        return matrices, scalars, prediction_stats
    
    def trace_of_cov(self, g):
        g_mean = torch.mean(g, dim=0, keepdim=True)
        g -= g_mean
        tr = torch.mean(g.norm(dim=1)**2)
        return tr.item()
    
    # def matrix_rank(self, g, hermitian=False):
    #     pass
        # rank = np.linalg.matrix_rank(g.detach().data.cpu().numpy(), hermitian=hermitian).astype(float).mean()
        # return rank
        # return torch.linalg.matrix_rank(g, hermitian=hermitian).float().mean().item()
    
    def prepare_variables(self, ft_per_sample_grads, concatenated_weights, scalars, tag, ind: str = None):
        # if 'weight' in tag:
        #     scalars[f'cumm_gradients_rank_{ind}'][tag] = min(ft_per_sample_grads[tag].shape[1:])
        #     scalars[f'rank_of_gradients_{ind}'][tag] = self.matrix_rank(ft_per_sample_grads[tag]) / min(ft_per_sample_grads[tag].shape[1:]) # ???
        ft_per_sample_grads[tag] = ft_per_sample_grads[tag].reshape(ft_per_sample_grads[tag].shape[0], -1)
        normed_ft_per_sample_grad = ft_per_sample_grads[tag] / (1e-9 + torch.norm(ft_per_sample_grads[tag], dim=1, keepdim=True))
        concatenated_weights = torch.cat((concatenated_weights, ft_per_sample_grads[tag]), dim=1)
        # scalars[f'trace_of_cov_{ind}'][tag] = self.trace_of_cov(ft_per_sample_grads[tag])
        return normed_ft_per_sample_grad, concatenated_weights
    
    def prepare_concatenated_weights(self, ft_per_sample_grads, concatenated_weights, scalars, ind: str = None):
        # scalars[f'rank_of_gradients_{ind}']['concatenated_weights'] = sum(scalars[f'rank_of_gradients_{ind}'][tag] * scalars[f'cumm_gradients_rank_{ind}'][tag] for tag in scalars[f'rank_of_gradients_{ind}']) / sum(scalars[f'cumm_gradients_rank_{ind}'].values())
        ft_per_sample_grads['concatenated_weights'] = concatenated_weights
        normed_concatenated_weights = ft_per_sample_grads['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads['concatenated_weights'], dim=1, keepdim=True))
        scalars[f'trace_of_cov_{ind}']['concatenated_weights'] = self.trace_of_cov(ft_per_sample_grads['concatenated_weights'])
        # scalars[f'gradients_subspace_dim_{ind}']['concatenated_weights'] = self.matrix_rank(normed_concatenated_weights)
        return normed_concatenated_weights
    
    def gather_metrics(self, ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag, hermitian=False, ind: str = None):
        matrices[f'similarity_{ind}'][tag] = normed_ft_per_sample_grad1 @ normed_ft_per_sample_grad2.T
        matrices[f'graham_{ind}'][tag] = ft_per_sample_grads1[tag] @ ft_per_sample_grads2[tag].T / matrices[f'similarity_{ind}'][tag].shape[0]
        matrices[f'cov_{ind}'][tag] = (ft_per_sample_grads1[tag] - ft_per_sample_grads1[tag].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads2[tag] - ft_per_sample_grads2[tag].mean(dim=0, keepdim=True)).T / matrices[f'similarity_{ind}'][tag].shape[0]
        # scalars[f'rank_of_similarity_{ind}'][tag] = self.matrix_rank(matrices[f'similarity_{ind}'][tag], hermitian=hermitian)
        if ind != '12':
            matrices[f'similarity_{ind}'][tag] = (matrices[f'similarity_{ind}'][tag] + matrices[f'similarity_{ind}'][tag].T) / 2
            matrices[f'graham_{ind}'][tag] = (matrices[f'graham_{ind}'][tag] + matrices[f'graham_{ind}'][tag].T) / 2
            matrices[f'cov_{ind}'][tag] = (matrices[f'cov_{ind}'][tag] + matrices[f'cov_{ind}'][tag].T) / 2
    
            
# wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
class Stiffness(torch.nn.Module):
    # add option to compute loss directly
    # add option with train-val
    def __init__(self, model, num_classes, x_true1, y_true1, logger=None, every_nb=1, x_true2=None, y_true2=None):
        super().__init__()
        self.per_sample_grad = PerSampleGrad(model)
        self.num_classes = num_classes
        idxs = torch.arange(0, x_true1.shape[0], every_nb)
        self.x_true1 = x_true1[idxs]
        self.y_true1 = y_true1[idxs]
        self.x_true2 = x_true2[idxs] if x_true2 is not None else None
        self.y_true2 = y_true2[idxs] if y_true2 is not None else None
        self.logger = logger
        
    def log_stiffness(self, step):
        stifness_heatmaps = {}
        stiffness_logs = {}
        stiffness_hists = {}
        matrices, scalars, prediction_stats = self.forward(self.x_true1, self.y_true1, self.x_true2, self.y_true2)
        
        # for tag in matrices[0]:
        #     # stiffness_logs[f'stiffness_sharpness/{tag}'] = self.sharpness(matrices[0][tag]['concatenated_weights'])
        #     # fig, ax  = plt.subplots(1, 1, figsize=(10, 10))
        #     # set a heatmap pallette to red-white-blue
        #     # stifness_heatmaps[f'stiffness/{tag}'] = sns.heatmap(matrices[0][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
        #     # plt.close(fig)
        #     if 'similarity' in tag and '12' not in tag:
        #         labels_true = self.y_true1.cpu().numpy() if '11' in tag else self.y_true2.cpu().numpy()
        #         labels_pred, unsolicited_ratio = self.clustering(matrices[0][tag]['concatenated_weights'], labels_true)
        #         acc = (labels_pred == labels_true).sum() / labels_true.shape[0]
        #         stiffness_logs[f'clustering/accuracy_{tag}'] = acc
        #         stiffness_logs[f'clustering/unsolicited_ratio_{tag}'] = unsolicited_ratio
        #         # stiffness_hists[f'clustering/histogram_{tag}'] = labels_pred  
        
        # for tag in matrices[1]:
            # fig, ax  = plt.subplots(1, 1, figsize=(10, 10))
            # stifness_heatmaps[f'class_stiffness/{tag}'] = sns.heatmap(matrices[1][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
            # plt.close(fig)
                   
        for tag in scalars[0]:
            stiffness_logs[f'traces of covs & ranks/{tag}'] = scalars[0][tag]['concatenated_weights']
            
        for tag in scalars[1]:
            stiffness_logs[f'stiffness/{tag}'] = scalars[1][tag]['concatenated_weights']
            
        for tag in prediction_stats:
            stiffness_logs[f'prediction_stats/{tag}'] = prediction_stats[tag]
        
        stiffness_logs['steps/stiffness_train'] = step
        
        # self.logger.log_figures(stifness_heatmaps, step)
        self.logger.log_scalars(stiffness_logs, step)
        # self.logger.log_histogram(stiffness_hists, step)
        
        
    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        matrices = defaultdict(dict)
        scalars = defaultdict(dict)
        matrices[0], scalars[0], prediction_stats = self.per_sample_grad(x_true1, y_true1, x_true2, y_true2) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
        scalars[1]['expected_stiffness_cosine_11'] = self.cosine_stiffness(matrices[0]['similarity_11']) 
        scalars[1]['expected_stiffness_sign_11'] = self.sign_stiffness(matrices[0]['similarity_11']) 
        matrices[1]['c_stiffness_cosine_11'], scalars[1]['stiffness_between_classes_cosine_11'], scalars[1]['stiffness_within_classes_cosine_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=False)
        matrices[1]['c_stiffness_sign_11'], scalars[1]['stiffness_between_classes_sign_11'], scalars[1]['stiffness_within_classes_sign_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=True)
        if x_true2 is not None:
            scalars[1]['expected_stiffness_cosine_12'] = self.cosine_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_sign_12'] = self.sign_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_cosine_22'] = self.cosine_stiffness(matrices[0]['similarity_22']) 
            scalars[1]['expected_stiffness_sign_22'] = self.sign_stiffness(matrices[0]['similarity_22']) 
            matrices[1]['c_stiffness_cosine_12'], scalars[1]['stiffness_between_classes_cosine_12'], scalars[1]['stiffness_within_classes_cosine_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_12'], scalars[1]['stiffness_between_classes_sign_12'], scalars[1]['stiffness_within_classes_sign_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=True)
            matrices[1]['c_stiffness_cosine_22'], scalars[1]['stiffness_between_classes_cosine_22'], scalars[1]['stiffness_within_classes_cosine_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_22'], scalars[1]['stiffness_between_classes_sign_22'], scalars[1]['stiffness_within_classes_sign_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=True)
            scalars[1]['expected_stiffness_diagonal_cosine_12'], scalars[1]['expected_stiffness_diagonal_sign_12'] = self.diagonal_instance_stiffness(matrices[0]['similarity_12'])
        return matrices, scalars, prediction_stats
    
    def cosine_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def sign_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(torch.sign(v)) - torch.diagonal(torch.sign(v)).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def class_stiffness(self, similarity_matrices, y_true, whether_sign=False):
        c_stiffness = {}
        # extract the indices into dictionary from y_true tensor where the class is the same
        indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
        indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
        for k, similarity_matrix in similarity_matrices.items():
            c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
            for c1, idxs1 in indices.items():
                for c2, idxs2 in indices.items():
                    sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
                    sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
                    c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
        stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
        stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
        return c_stiffness, stiffness_between_classes, stiffness_within_classes  
    
    def diagonal_instance_stiffness(self, similarity_matrices):
        expected_diag_stiffness_cosine = {k: torch.mean(torch.diag(v)).item() for k, v in similarity_matrices.items()}
        expected_diag_stiffness_sign = {k: torch.mean(torch.sign(torch.diag(v))).item() for k, v in similarity_matrices.items()}
        return expected_diag_stiffness_cosine, expected_diag_stiffness_sign
        
    def sharpness(self, similarity_matrix):
        w, _ = torch.linalg.eig(similarity_matrix)
        max_eig = torch.max(w.real) # .abs()??
        return max_eig.item()
    
    def clustering(self, similarity_matrix, labels_true):
        similarity_matrix_ = similarity_matrix.cpu().numpy()
        labels_pred = SpectralClustering(n_clusters=self.num_classes, affinity='precomputed', n_init=100, assign_labels='discretize').fit_predict((1+similarity_matrix_)/2)
        labels_pred, unsolicited_ratio = self.retrieve_info(labels_pred, labels_true)
        return labels_pred, unsolicited_ratio
    
    
    def retrieve_info(self, cluster_labels, y_train):
        ## ValueError: attempt to get argmax of an empty sequence: dist.argmax()
        # Initializing
        unsolicited_ratio = 0.0
        denominator = 0.0
        reference_labels = {}
        # For loop to run through each label of cluster label
        for label in range(len(np.unique(y_train))):
            index = np.where(cluster_labels==label, 1, 0)
            dist = np.bincount(y_train[index==1])
            num = dist.argmax()
            unsolicited_ratio += (dist.sum() - dist.max())
            denominator += dist.sum()
            reference_labels[label] = num
        proper_labels = [reference_labels[label] for label in cluster_labels]
        proper_labels = np.array(proper_labels)
        unsolicited_ratio /= denominator
        return proper_labels, unsolicited_ratio
    
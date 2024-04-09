from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
import torch


def variance_angle_pairwise(x_data, y_true, evaluators, label, phase): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        internal_repr /= (torch.norm(internal_repr, dim=1, keepdim=True) + eps)
        # for k in range(len(internal_repr)):
        #     internal_repr[k] /= (torch.norm(internal_repr[k]) + eps)
            
        cos = internal_repr @ internal_repr.T
        abs_angle_matrix = torch.abs(torch.acos(cos))  # abs dlatego że chodzi o absolutne kąty
        
        general_mean = torch.triu(abs_angle_matrix, diagonal=1)  # wybieram tak by kąty się nie powtarzały
        general_mean = general_mean[general_mean.nonzero(as_tuple=True)].flatten().mean()  # mi_g
        
        for i in range(denom_class):
            idxs_i = np.where(y_true.cpu().numpy() == i)[0]
            sub_matrix = abs_angle_matrix[idxs_i][:, idxs_i]  # kąt od wszystkich z danej klasy do wszystkich z danej klasy 
            sub_matrix = torch.triu(sub_matrix, diagonal=1)
            sub_vector = sub_matrix[sub_matrix.nonzero(as_tuple=True)].flatten()  # [alpha_i,c]_i, i - każdy kąt między gradientami jednej klasy
            off_diag_mean = sub_vector.mean()  # mi_c
            within_class_cov += (sub_vector - off_diag_mean).pow(2).mean().item()  # mean_i(alpha_i,c - mi_c)**2
            between_class_cov += (off_diag_mean - general_mean).pow(2).item()  # sum_c(mi_c - mi_g)

        within_class_cov /= denom_class
        between_class_cov /= denom_class
        total_class_cov = within_class_cov + between_class_cov
        
        evaluators[f'within_cov_abs_angle_{label}_{phase}/{name}'] = within_class_cov
        evaluators[f'between_cov_abs_angle_{label}_{phase}/{name}'] = between_class_cov
        evaluators[f'total_cov_abs_angle_{label}_{phase}/{name}'] = total_class_cov
        
        normalized_wcc = within_class_cov / (total_class_cov + eps) 
        normalized_bcc = between_class_cov / (total_class_cov + eps)
        
        
        evaluators[f'within_cov_abs_angle_normalized_{label}_{phase}/{name}'] = normalized_wcc
        evaluators[f'between_cov_abs_angle_normalized_{label}_{phase}/{name}'] = normalized_bcc
    return evaluators


def gradient_direction_variance(gradients_per_params, evaluators, label, phase):
    eps = torch.finfo(torch.float32).eps
    for param_name, gradients in gradients_per_params.items():
        gradients /= (torch.norm(gradients, dim=1, keepdim=True) + eps)
            
        cos = gradients @ gradients.T
        
        unique_cos = torch.triu(cos, diagonal=1)  # wybieram tak by kąty się nie powtarzały
        unique_cos = unique_cos[unique_cos.nonzero(as_tuple=True)].flatten()
        gdv = (1 - unique_cos.mean().item()) / 2
        
        evaluators[f'gradient_direction_variance_{label}_{phase}/{param_name}'] = gdv
    return evaluators


def gradient_direction_error(gradients_per_params, evaluators, label, phase):
    eps = torch.finfo(torch.float32).eps
    for param_name, gradients in gradients_per_params.items():
        step_gradient = gradients.mean(axis=0, keepdim=True)
        step_gradient /= (torch.norm(step_gradient, dim=1, keepdim=True) + eps)
        gradients /= (torch.norm(gradients, dim=1, keepdim=True) + eps)
        
        cos = gradients @ step_gradient.T
        gde = (1 - cos.mean().item()) / 2
        evaluators[f'gradient_direction_error_{label}_{phase}/{param_name}'] = gde


def gradient_ssr_and_largest_eigenvalue(gradients_per_params, evaluators, label, phase, cutoff=1000): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    for param_name, gradients in gradients_per_params.items():
        gramian = gradients @ gradients.T    # (N, N)
        singular_squared = torch.linalg.eig(gramian)[0].float()
        square_stable_rank = singular_squared.sum() / max(singular_squared)
        evaluators[f'square_stable_rank_gradients_{label}_{phase}/{param_name}'] = square_stable_rank.item()
        evaluators[f'the_largest_eigenvalue_gradients_{label}_{phase}/{param_name}'] = max(singular_squared).item()
        
        
        gradients /= (torch.norm(gradients, dim=1, keepdim=True) + eps)  # (N, D)
        
        gramian = gradients @ gradients.T    # (N, D)
        singular_squared = torch.linalg.eig(gramian)[0].float()
        square_stable_rank = singular_squared.sum() / max(singular_squared)
        evaluators[f'square_stable_rank_gradients_normalized_{label}_{phase}/{param_name}'] = square_stable_rank.item()
        evaluators[f'the_largest_eigenvalue_gradients_normalized_{label}_{phase}/{param_name}'] = max(singular_squared).item()
    return evaluators


def variance_with_gradient_general_mean_and_class_mean(gradients_per_params, y_true, evaluators, label, phase): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    y_true = y_true.cpu().numpy()
    classes = np.unique(y_true)
    denom_class = classes.shape[0]
    for param_name, gradients in gradients_per_params.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        total_class_cov = 0.0
        
        general_mean = gradients.mean(axis=0, keepdim=True)
        general_mean /= (torch.norm(general_mean, dim=1, keepdim=True) + eps)
        
        for i in range(denom_class):
            idxs_i = np.where(y_true == i)[0]
            sub_gradients = gradients[idxs_i]
            class_mean = sub_gradients.mean(axis=0, keepdim=True)
            class_mean /= (torch.norm(class_mean, dim=1, keepdim=True) + eps)
            sub_gradients /= (torch.norm(sub_gradients, dim=1, keepdim=True) + eps)
            within_class_cov += ((1 - class_mean @ sub_gradients.T)/2).pow(2).mean().item()
            between_class_cov += ((1 - class_mean @ general_mean.T)/2).pow(2).item()
            total_class_cov += ((1 - general_mean @ sub_gradients.T)/2).pow(2).mean().item()

        within_class_cov /= denom_class
        between_class_cov /= denom_class
        total_class_cov /= denom_class
        
        evaluators[f'within_class_gradient_disparity_{label}_{phase}/{param_name}'] = within_class_cov
        evaluators[f'between_class_gradient_disparity_{label}_{phase}/{param_name}'] = between_class_cov
        evaluators[f'total_class_gradient_disparity_{label}_{phase}/{param_name}'] = total_class_cov
        
        normalized_wcc = within_class_cov / (total_class_cov + eps) 
        normalized_bcc = between_class_cov / (total_class_cov + eps)
        
        
        evaluators[f'within_class_gradient_disparity_normalized_{label}_{phase}/{param_name}'] = normalized_wcc
        evaluators[f'between_class_gradient_disparity_normalized_{label}_{phase}/{param_name}'] = normalized_bcc
    return evaluators


def variance_eucl(x_data, y_true, evaluators, label, phase, batch_size=200):
    eps = torch.finfo(torch.float32).eps
    y_true = y_true.cpu()
    classes = np.unique(y_true.numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr, dim=0, keepdim=True)  # (1, D)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c]  # (N_c, D)
            class_mean = torch.mean(class_internal_repr, dim=0, keepdim=True)  # (1, D)
            class_internal_repr_sub = class_internal_repr - class_mean  # (N_c, D)
            # print(class_internal_repr_sub.shape)
            # for sample in class_internal_repr_sub:
            #     within_class_cov += sample.unsqueeze(1) @ sample.unsqueeze(0)
            # within_class_cov = (class_internal_repr_sub.unsqueeze(2) @ class_internal_repr_sub.unsqueeze(1)).mean(dim=0)  # (N_c, D, 1) x (N_c, 1, D) -> (D, D)
            
            S = batch_size # depends on GPU available (GB), size of a batch
            K = class_internal_repr_sub.shape[0] // S
            within_class_cov += sum([(class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(2) @ class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(1)).mean(dim=0) for i in range(K)]) / K  # (N_c, D, 1) x (N_c, 1, D) -> (D, D)
            between_sample = (class_mean - general_mean)
            between_class_cov += between_sample.T @ between_sample  # (D, 1) x (1, D) -> (D, D)
        
        within_class_cov /= denom_class  # (D, D)
        between_class_cov /= denom_class  # (D, D)
        total_class_cov = within_class_cov + between_class_cov  # (D, D)
        
        trace_wcc = torch.trace(within_class_cov)
        trace_bcc = torch.trace(between_class_cov)
        trace_tcc = torch.trace(total_class_cov)
        
        # evaluators[f'wwc_square_stable_rank_{label}_{phase}/{name}'] = trace_wcc.item()
        # evaluators[f'bcc_square_stable_rank_{label}_{phase}/{name}'] = trace_bcc.item()
        # evaluators[f'tcc_square_stable_rank_{label}_{phase}/{name}'] = trace_tcc.item()
        
        normalized_wcc = trace_wcc / (trace_tcc + eps) 
        normalized_bcc = trace_bcc / (trace_tcc + eps)
        
        evaluators[f'within_cov_eucl_normalized_{label}_{phase}/{name}'] = normalized_wcc.item()
        evaluators[f'between_cov_eucl_normalized_{label}_{phase}/{name}'] = normalized_bcc.item()
        
        
        # rank_wcc = torch.linalg.matrix_rank(within_class_cov)
        # rank_bcc = torch.linalg.matrix_rank(between_class_cov)
        # rank_tcc = torch.linalg.matrix_rank(total_class_cov)
        
        # evaluators[f'within_cov_rank_{label}/{name}'] = rank_wcc.item()
        # evaluators[f'between_cov_rank/{name}'] = rank_bcc.item()
        # evaluators[f'total_cov_rank/{name}'] = rank_tcc.item()
        
        # A = within_class_cov.T @ within_class_cov
        # square_stable_rank_wcc = torch.trace(A) #/ eigh(A.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(A, k=1)[0][0]
        # B = between_class_cov.T @ between_class_cov
        # square_stable_rank_bcc = torch.trace(B) #/ eigh(B.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(B, k=1)[0][0]
        # C = total_class_cov.T @ total_class_cov
        # square_stable_rank_tcc = torch.trace(C) #/ eigh(C.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(C, k=1)[0][0]
        
        # evaluators[f'within_cov_square_stable_rank/{name}'] = square_stable_rank_wcc.item()
        # evaluators[f'between_cov_square_stable_rank/{name}'] = square_stable_rank_bcc.item()
        # evaluators[f'total_cov_square_stable_rank/{name}'] = square_stable_rank_tcc.item()
        
        
import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
    
class GradientsSpectralStiffness(torch.nn.Module):
    # compute loss and grad per sample 
    # minusem normowania gradientów jest uważanie ich za równie istotnych
    def __init__(self, held_out, model, cutoff):
        super().__init__()
        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out['proper_x_left']
        self.held_out_proper_x_right = held_out['proper_x_right']
        self.held_out_blurred_x_right = held_out['blurred_x_right']
        self.held_out_y = held_out['y']
        self.model = model
        # self.cutoff = cutoff
        self.criterion = torch.nn.CrossEntropyLoss()#.to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.eps = torch.finfo(torch.float32).eps
        self.logger = None
        self.subsampling = defaultdict(lambda: None)
        
    def compute_loss(self, params, buffers, config, sample, target):
        batch0 = sample[0].unsqueeze(0)
        batch1 = sample[1].unsqueeze(0)
        targets = target.unsqueeze(0)
        kwargs = {'left_branch_intervention': config.extra['left_branch_intervention'],
                  'right_branch_intervention': config.extra['right_branch_intervention'],
                  'enable_left_branch': config.extra['enable_left_branch'],
                  'enable_right_branch': config.extra['enable_right_branch']}
        y_pred = functional_call(self.model, (params, buffers), (batch0, batch1), kwargs=kwargs)
        loss = self.criterion(y_pred, targets)
        return loss, y_pred

    def forward(self, step, config, scope, phase, kind):
        self.model.eval()
        x_true1 = self.held_out_proper_x_left
        x_true2 = self.held_out_proper_x_right if kind == 'proper' else self.held_out_blurred_x_right
        y_true = self.held_out_y
     
        classes = np.unique(y_true.cpu().numpy())
        prefix = lambda a, b, c : f'{a}_{b}_{c}_branch_{phase}'
        postfix = f'____{scope}____{phase}'
        chunk_size = 250

        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names and 'bias' not in k and 'downsample' not in k}
        params1 = {n: p for n, p in params.items() if 'left_branch' in n}
        params2 = {n: p for n, p in params.items() if 'right_branch' in n}
        params3 = {n: p for n, p in params.items() if 'main_branch' in n}
        buffers = {}
        evaluators = defaultdict(float)
        
        ft_per_sample_grads1, y_pred = None, None
        
        for i in range(y_true.shape[0] // chunk_size):  # accumulate grads
            ft_per_sample_grads1_, y_pred_ = self.ft_criterion(params1, buffers, config, (x_true1[i*chunk_size: (i+1)*chunk_size], x_true2[i*chunk_size: (i+1)*chunk_size]), y_true[i*chunk_size: (i+1)*chunk_size])
            ft_per_sample_grads1_ = {k1: v.detach().data for k1, v in ft_per_sample_grads1_.items()}
            self.prepare_gradients(ft_per_sample_grads1_)
            # self.sample_cordinates(ft_per_sample_grads1_)  # czy na pewno w róznym czasie chce samplować dla różnych partii danych?
            ft_per_sample_grads1, y_pred = self.update(ft_per_sample_grads1, y_pred, ft_per_sample_grads1_, y_pred_)
        
        ft_per_sample_grads2, y_pred = None, None
        
        for i in range(y_true.shape[0] // chunk_size):  # accumulate grads
            ft_per_sample_grads2_, y_pred_ = self.ft_criterion(params2, buffers, config, (x_true1[i*chunk_size: (i+1)*chunk_size], x_true2[i*chunk_size: (i+1)*chunk_size]), y_true[i*chunk_size: (i+1)*chunk_size])
            ft_per_sample_grads2_ = {k1: v.detach().data for k1, v in ft_per_sample_grads1_.items()}
            self.prepare_gradients(ft_per_sample_grads2_)
            # self.sample_cordinates(ft_per_sample_grads2_)  # czy na pewno w róznym czasie chce samplować dla różnych partii danych?
            ft_per_sample_grads2, y_pred = self.update(ft_per_sample_grads2, y_pred, ft_per_sample_grads2_, y_pred_)
        y_pred_label = torch.argmax(y_pred.data.squeeze(), dim=1)
        
        
        for c in classes:
            idxs_mask = y_pred_label == c
            evaluators[f'misclassification_per_class/{c}{postfix}'] = (y_pred_label[idxs_mask] != y_true[idxs_mask]).float().mean().item()
            
        del ft_per_sample_grads1_
        del ft_per_sample_grads2_
        del y_pred_label
            
        concatenated_grads = torch.empty((y_true.shape[0], 0), device=y_true.device)
        self.prepare_gradients(ft_per_sample_grads1, concatenated_grads)
        concatenated_grads = torch.empty((y_true.shape[0], 0), device=y_true.device)
        self.prepare_gradients(ft_per_sample_grads2, concatenated_grads)
        
        # self.prepare_variables(per_sample_grads, concatenated_grads)
        # self.sample_feats(per_sample_grads)
            
        # variance_angle_pairwise(ft_per_sample_grads1, y_true, evaluators, label='left_branch', phase=phase)
        # variance_angle_pairwise(ft_per_sample_grads2, y_true, evaluators, label='right_branch', phase=phase)
        gradient_direction_variance(ft_per_sample_grads1, evaluators, label='left_branch', phase=phase)
        gradient_direction_variance(ft_per_sample_grads2, evaluators, label='right_branch', phase=phase)
        gradient_direction_error(ft_per_sample_grads1, evaluators, label='left_branch', phase=phase)
        gradient_direction_error(ft_per_sample_grads2, evaluators, label='right_branch', phase=phase)
        gradient_ssr_and_largest_eigenvalue(ft_per_sample_grads1, evaluators, label='left_branch', phase=phase)
        gradient_ssr_and_largest_eigenvalue(ft_per_sample_grads2, evaluators, label='right_branch', phase=phase)
        variance_with_gradient_general_mean_and_class_mean(ft_per_sample_grads1, y_true, evaluators, label='left_branch', phase=phase)
        variance_with_gradient_general_mean_and_class_mean(ft_per_sample_grads2, y_true, evaluators, label='right_branch', phase=phase)
        
        # evaluators = self.prepare_and_calculate_ranks(ft_per_sample_grads1, evaluators, prefix('nonnormalized', 'feature'), postfix, to_normalize=False, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(ft_per_sample_grads2, evaluators, prefix('nonnormalized', 'feature'), postfix, to_normalize=False, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'batch'), postfix, normalize=False, batch_first=True, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'feature'), postfix, normalize=True, batch_first=False, risky_names=('concatenated_grads'))
        evaluators = self.prepare_and_calculate(ft_per_sample_grads1, y_true, evaluators, prefix('normalized', 'batch_first', 'left'), postfix, to_normalize=True, batch_first=True, risky_names=('concatenated_grads'))
        evaluators = self.prepare_and_calculate(ft_per_sample_grads2, y_true, evaluators, prefix('normalized', 'batch_first', 'right'), postfix, to_normalize=True, batch_first=True, risky_names=('concatenated_grads'))
        
        
        self.model.train()
        evaluators[f'steps/tunnel_grads_{phase}'] = step
        self.logger.log_scalars(evaluators, step)
        
        
    def prepare_gradients(self, per_sample_grads, concatenated_grads=None): # sampluj gdy za duże
        for param_name in per_sample_grads:
            per_sample_grads[param_name] = per_sample_grads[param_name].reshape(per_sample_grads[param_name].shape[0], -1)
            if concatenated_grads is not None:
                concatenated_grads = torch.cat((concatenated_grads, per_sample_grads[param_name]), dim=1)
        if concatenated_grads is not None:
            per_sample_grads['concatenated_grads'] = concatenated_grads
        
    # def sample_cordinates(self, per_sample_grads):
    #     for name in per_sample_grads:
    #         per_sample_grads[name] = self.adjust_representation(per_sample_grads[name])
    
    # def adjust_representation(self, grads):
    #     subsampling = torch.randperm(grads.size(1))[:self.cutoff].sort()[0]
    #     representation = torch.index_select(grads, 1, subsampling.to(self.device))
    #     return representation
    
    # def sample_feats_old(self, per_sample_grads):
    #     for name in per_sample_grads:
    #         if name in self.subsampling:
    #             per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
    #         elif per_sample_grads[name].size(1) > self.cutoff:
    #             self.subsampling[name] = torch.randperm(per_sample_grads[name].size(1))[:self.cutoff].sort()[0]
    #             per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
                
    # def adjust_representation_old(self, grads, name):
    #     representation = torch.index_select(grads, 1, self.subsampling[name].to(self.device))
    #     return representation
    
    
    def update(self, per_sample_grads_old, y_pred_old, per_sample_grads_new, y_pred_new):
        if per_sample_grads_old is None:
            return per_sample_grads_new, y_pred_new
        for name in per_sample_grads_old:
            per_sample_grads_old[name] = torch.cat((per_sample_grads_old[name], per_sample_grads_new[name]), dim=0)
        y_pred_old = torch.cat((y_pred_old, y_pred_new), dim=0)
        return per_sample_grads_old, y_pred_old
    
    def prepare_and_calculate(self, gradients_per_params, y_true, evaluators, prefix, postfix, to_normalize=False, batch_first=True, risky_names=()):
        similarity_matrices = {}
        for param_name, gradients in gradients_per_params.items():
            if param_name in risky_names and not batch_first:  # bo wtedy za dużo obliczeń
                continue
            if to_normalize:
                gradients = gradients / (torch.norm(gradients, dim=1, keepdim=True) + self.eps)
            gradients = self.prepare_matrix(gradients, batch_first)
            similarity_matrices[param_name] = gradients
            ranks = self.calculate_rank(gradients)
            denom = gradients.size(0) if batch_first else gradients.T.size(0)
            
            name_dict = f'ranks_grads_{prefix}/{param_name}{postfix}'
            name_dict_ratio = f'ranks_grads_{prefix}_ratio/{param_name}{postfix}'
            evaluators[name_dict] = ranks
            evaluators[name_dict_ratio] = ranks / denom # check if dim makes sense
            # name_dict_null = f'{prefix}_null/{name}{postfix}'
            # evaluators[name_dict_null] = denom - ranks[0]
            
            # name_dict_square_stable = f'{prefix}_square_stable/{name}{postfix}'
            # name_dict_ratio_square_stable = f'{prefix}_ratio_square_stable/{name}{postfix}'
            # name_dict_null_square_stable = f'{prefix}_null_square_stable/{name}{postfix}'
            # evaluators[name_dict_square_stable] = ranks[1]
            # evaluators[name_dict_ratio_square_stable] = ranks[1] / denom # check if dim makes sense
            # evaluators[name_dict_null_square_stable] = denom - ranks[1]
        evaluators = self.cosine_stiffness(similarity_matrices, evaluators, prefix, postfix)
        evaluators = self.sign_stiffness(similarity_matrices, evaluators, prefix, postfix)
        evaluators = self.class_stiffness(similarity_matrices, y_true, evaluators, prefix, postfix)
        return evaluators
    
    
    def prepare_matrix(self, matrix, batch_first):
        matrix = matrix if batch_first else matrix.T
        # gramian_matrix = torch.cov(matrix)
        gramian_matrix = matrix @ matrix.T
        return gramian_matrix
    
    def calculate_rank(self, matrix): # jedyny pomysł to z paddingiem macierzy do maksymalnej
        rank = torch.linalg.matrix_rank(matrix).item()
        # square_stable_rank = (torch.diag(gramian_matrix).sum()).item() #/ torch.lobpcg(gramian_matrix, k=1)[0][0]).item()
        return rank
    
    
    def cosine_stiffness(self, similarity_matrices, evaluators, prefix, postfix):
        expected_stiffness = {f'cosine_stiffness_{prefix}/{k}{postfix}': ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        evaluators = evaluators | expected_stiffness
        return evaluators
    
    def sign_stiffness(self, similarity_matrices, evaluators, prefix, postfix):
        expected_stiffness = {f'sign_stiffness_{prefix}/{k}{postfix}': ((torch.sum(torch.sign(v)) - torch.diagonal(torch.sign(v)).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        evaluators = evaluators | expected_stiffness
        return evaluators
    
    def class_stiffness(self, similarity_matrices, y_true, evaluators, prefix, postfix, whether_sign=False):
        c_stiffness = {}
        classes = np.unique(y_true.cpu().numpy())
        num_classes = len(classes)
        # extract the indices into dictionary from y_true tensor where the class is the same
        indices = {c: torch.where(y_true == c)[0] for c in classes}
        indices = {c: t for c, t in indices.items() if t.shape[0] > 0}
        for k, similarity_matrix in similarity_matrices.items():
            c_stiffness[k] = torch.zeros((num_classes, num_classes), device=y_true.device)
            for c1, idxs1 in indices.items():
                for c2, idxs2 in indices.items():
                    sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
                    sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
                    c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
        stiffness_between_classes = {f'stiffness_between_classes_{prefix}/{k}{postfix}': ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
        stiffness_within_classes = {f'stiffness_within_classes_{prefix}/{k}{postfix}': (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        evaluators = evaluators | stiffness_between_classes | stiffness_within_classes
        return evaluators
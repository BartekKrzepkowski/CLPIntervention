from abc import abstractmethod
from collections import defaultdict
import json
import os
from pathlib import Path

import torch
from torch.func import functional_call, grad, vmap

from src.modules.aux_modules_collapse import variance_eucl
from src.utils.utils_optim import FORBIDDEN_LAYER_TYPES, get_every_but_forbidden_parameter_names


class TraceFIM(torch.nn.Module):
    # Empirical Fisher probe with common label draws and an isolated RNG.

    def __init__(
        self, held_out, model, num_classes, postfix, m_sampling=1,
        sampling_seed=0, chunk_size=16,
    ):
        super().__init__()
        if m_sampling < 1:
            raise ValueError("m_sampling must be positive")
        if int(chunk_size) < 1:
            raise ValueError("FIM chunk_size must be positive")

        self.device = next(model.parameters()).device
        self.held_out_proper_x_left = held_out["proper_x_left"]
        self.held_out_proper_x_right = held_out["proper_x_right"]
        self.held_out_blurred_x_right = held_out["blurred_x_right"]
        sizes = {
            self.held_out_proper_x_left.size(0),
            self.held_out_proper_x_right.size(0),
            self.held_out_blurred_x_right.size(0),
        }
        if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
            raise ValueError("held-out modality tensors must have the same non-zero batch size")

        self.model = model
        self.num_classes = num_classes
        self.postfix = postfix
        self.m_sampling = m_sampling
        self.sampling_seed = sampling_seed
        self.chunk_size = int(chunk_size)
        self.logger = None
        self.artifact_path = None
        self.penalized_parameter_names = set(
            get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        )
        self.per_sample_trace = vmap(
            self._grad_and_trace,
            in_dims=(None, None, None, 0),
            randomness="different",
        )

    @staticmethod
    def _forward_kwargs(config):
        return {
            "left_branch_intervention": config.extra["left_branch_intervention"],
            "right_branch_intervention": config.extra["right_branch_intervention"],
            "enable_left_branch": config.extra["enable_left_branch"],
            "enable_right_branch": config.extra["enable_right_branch"],
        }

    def _compute_loss(self, params, buffers, config, sample):
        x_left, x_right, sampled_target = sample
        logits = functional_call(
            self.model,
            (params, buffers),
            (x_left.unsqueeze(0), x_right.unsqueeze(0)),
            kwargs=self._forward_kwargs(config),
        )
        return torch.nn.functional.cross_entropy(logits, sampled_target.unsqueeze(0))

    def _grad_and_trace(self, params, buffers, config, sample):
        sample_grads = grad(self._compute_loss)(params, buffers, config, sample)
        return {name: gradient.square().sum() for name, gradient in sample_grads.items()}

    def _sample_targets(self, x_left, x_right, config, global_step):
        with torch.no_grad():
            logits = self.model(x_left, x_right, **self._forward_kwargs(config))
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.sampling_seed + int(global_step))
        return [
            torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            for _ in range(self.m_sampling)
        ]

    def forward(self, global_step, config, kind):
        if kind not in {"proper", "blurred"}:
            raise ValueError(f"Unsupported FIM probe kind: {kind}")

        was_training = self.model.training
        self.model.eval()
        x_left = self.held_out_proper_x_left.to(self.device)
        x_right_source = (
            self.held_out_proper_x_right
            if kind == "proper"
            else self.held_out_blurred_x_right
        )
        x_right = x_right_source.to(self.device)
        params = {
            name: parameter.detach()
            for name, parameter in self.model.named_parameters()
            if name in self.penalized_parameter_names and parameter.requires_grad
        }
        branch_params = {
            "left": {name: value for name, value in params.items() if "left_branch" in name},
            "right": {name: value for name, value in params.items() if "right_branch" in name},
        }
        if not any(branch_params.values()):
            raise ValueError("TraceFIM requires a trainable left_branch or right_branch")
        buffers = {name: buffer.detach() for name, buffer in self.model.named_buffers()}
        evaluators = defaultdict(float)
        branch_traces = {"left": 0.0, "right": 0.0}
        branch_weight_traces = {"left": 0.0, "right": 0.0}
        branch_parameter_counts = {
            branch: sum(parameter.numel() for parameter in selected_params.values())
            for branch, selected_params in branch_params.items()
        }
        branch_weight_parameter_counts = {
            branch: sum(
                parameter.numel()
                for name, parameter in selected_params.items()
                if name.endswith("weight")
            )
            for branch, selected_params in branch_params.items()
        }

        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index or torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                sampled_targets = self._sample_targets(
                    x_left, x_right, config, global_step
                )
                sample_count = int(x_left.size(0))
                denominator = self.m_sampling * sample_count
                for targets in sampled_targets:
                    for start in range(0, sample_count, self.chunk_size):
                        stop = min(start + self.chunk_size, sample_count)
                        sample = (
                            x_left[start:stop],
                            x_right[start:stop],
                            targets[start:stop],
                        )
                        for branch, selected_params in branch_params.items():
                            if not selected_params:
                                continue
                            traces = self.per_sample_trace(
                                selected_params, buffers, config, sample
                            )
                            for parameter_name, per_example_trace in traces.items():
                                trace = (
                                    per_example_trace.detach().sum().item()
                                    / denominator
                                )
                                evaluators[
                                    f"trace_fim_{self.postfix}_{kind}/{parameter_name}"
                                ] += trace
                                branch_traces[branch] += trace
                                if parameter_name.endswith("weight"):
                                    branch_weight_traces[branch] += trace
        finally:
            self.model.train(was_training)

        left_trace = branch_weight_traces["left"]
        right_trace = branch_weight_traces["right"]
        prefix = f"trace_fim_overall_{self.postfix}"
        left_all_trace = branch_traces["left"]
        right_all_trace = branch_traces["right"]
        left_all_count = branch_parameter_counts["left"]
        right_all_count = branch_parameter_counts["right"]
        evaluators[f"{prefix}/{kind}_trace"] = left_all_trace + right_all_trace
        evaluators[f"{prefix}/{kind}_trace1"] = left_all_trace
        evaluators[f"{prefix}/{kind}_trace2"] = right_all_trace
        evaluators[f"{prefix}/{kind}_parameter_count1"] = left_all_count
        evaluators[f"{prefix}/{kind}_parameter_count2"] = right_all_count
        if left_all_count:
            evaluators[f"{prefix}/{kind}_trace1_per_parameter"] = (
                left_all_trace / left_all_count
            )
        if right_all_count:
            evaluators[f"{prefix}/{kind}_trace2_per_parameter"] = (
                right_all_trace / right_all_count
            )
        if left_all_count and right_all_count:
            evaluators[f"{prefix}/{kind}_ratio_left_to_right"] = (
                left_all_trace
                / (right_all_trace + torch.finfo(torch.float32).eps)
            )
        evaluators[f"{prefix}/{kind}_trace_per_parameter"] = (
            (left_all_trace + right_all_trace) / (left_all_count + right_all_count)
        )
        evaluators[f"{prefix}/{kind}_trace_weight"] = left_trace + right_trace
        evaluators[f"{prefix}/{kind}_trace1_weight"] = left_trace
        evaluators[f"{prefix}/{kind}_trace2_weight"] = right_trace
        left_count = branch_weight_parameter_counts["left"]
        right_count = branch_weight_parameter_counts["right"]
        evaluators[f"{prefix}/{kind}_parameter_count1_weight"] = left_count
        evaluators[f"{prefix}/{kind}_parameter_count2_weight"] = right_count
        if left_count:
            evaluators[f"{prefix}/{kind}_trace1_weight_per_parameter"] = (
                left_trace / left_count
            )
        if right_count:
            evaluators[f"{prefix}/{kind}_trace2_weight_per_parameter"] = (
                right_trace / right_count
            )
        evaluators[f"{prefix}/{kind}_trace_weight_per_parameter"] = (
            (left_trace + right_trace) / (left_count + right_count)
        )
        if left_count and right_count:
            evaluators[f"{prefix}/{kind}_ratio_left_to_right_weight"] = left_trace / (
                right_trace + torch.finfo(torch.float32).eps
            )
        evaluators[f"steps/trace_fim_{self.postfix}"] = global_step
        if self.artifact_path is not None:
            artifact_path = Path(self.artifact_path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "version": 1,
                "global_step": int(global_step),
                "phase": int(getattr(config, "phase", 0) or 0),
                "phase_epoch": int(
                    getattr(config, "active_phase_epoch", 0) or 0
                ),
                "kind": kind,
                "metrics": dict(evaluators),
            }
            with artifact_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.logger is not None:
            self.logger.log_scalars(evaluators, global_step)
        return dict(evaluators)


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
        was_training = self.model.training
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
        self.model.train(was_training)
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
        import matplotlib.pyplot as plt

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
        self.modules_list = [torch.nn.ReLU]
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
        if not self.nb_of_dead_relu:
            return
        number = sum([(self.nb_of_dead_relu[name] == max_dataset).sum() for name in self.nb_of_dead_relu]) / sum([self.nb_of_dead_relu[name].shape[0] for name in self.nb_of_dead_relu])
        evaluators = {f'nb_of_dead_relu_units_{"left" if self.is_left_branch else "right"}_branch/overall_frac____epoch____{phase}': number}
        numbers = [(self.nb_of_dead_relu[name] == max_dataset).float().mean() for name in self.nb_of_dead_relu]
        for i, name in enumerate(self.nb_of_dead_relu):
            evaluators[f'nb_of_dead_relu_units_{"left" if self.is_left_branch else "right"}_branch/{name}_frac____epoch____{phase}'] = numbers[i]
        evaluators[f'steps/dead_relu_{"left" if self.is_left_branch else "right"}'] = step
        self.logger.log_scalars(evaluators, step)
        self.nb_of_dead_relu = {}
        torch.cuda.empty_cache()

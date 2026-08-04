from collections import defaultdict

import torch


def source_variances(left_varying, right_varying, dim=0):
    """Return per-unit source variances for left- and right-varying inputs."""
    if left_varying.shape != right_varying.shape:
        raise ValueError("left- and right-varying responses must have equal shapes")
    if left_varying.size(dim) < 2:
        raise ValueError("RSV requires at least two augmentations per modality")

    left_variance = left_varying.var(dim=dim, unbiased=False)
    right_variance = right_varying.var(dim=dim, unbiased=False)
    return left_variance, right_variance


def relative_source_variance(left_varying, right_varying, dim=0):
    """Return Kleinman RSV: ``+1=left`` and ``-1=right``."""
    left_variance, right_variance = source_variances(
        left_varying, right_varying, dim=dim
    )
    denominator = left_variance + right_variance
    return torch.where(
        denominator > 0,
        (left_variance - right_variance) / denominator,
        torch.zeros_like(denominator),
    )

# czy w papierze chodziło o stale martwe neurony?
class DeadActivationCallback:
    def __init__(self):
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.idx = 0
        self.is_able = False

    def __call__(self, module, input, output):
        if isinstance(module, torch.nn.ReLU) and self.is_able:
            self.dead_acts[f'per_scalar_{module._get_name()}_{self.idx}'] += torch.sum(output == 0).item() # zrób .all względem wymiaru batcha
            self.denoms[f'per_scalar_{module._get_name()}_{self.idx}'] += output.numel()
            cond = torch.all(output == 0, dim=0) if len(output.shape) <= 2 else torch.all(output == 0, dim=0).all(dim=1).all(dim=1)
            self.dead_acts[f'per_neuron_{module._get_name()}_{self.idx}'] += torch.sum(cond).item()
            self.denoms[f'per_neuron_{module._get_name()}_{self.idx}'] += output.size(1)
            self.idx += 1
            
    def reset(self):
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.idx = 0
        
    def prepare_mean(self):
        self.dead_acts['per_scalar_total'] = sum(self.dead_acts[tag] for tag in self.dead_acts if 'per_scalar' in tag) # rozróżnienei na dwa totale
        self.denoms['per_scalar_total'] = sum(self.denoms[tag] for tag in self.denoms if 'per_scalar' in tag)
        self.dead_acts['per_neuron_total'] = sum(self.dead_acts[tag] for tag in self.dead_acts if 'per_neuron' in tag) # rozróżnienei na dwa totale
        self.denoms['per_neuron_total'] = sum(self.denoms[tag] for tag in self.denoms if 'per_neuron' in tag)
        new_dead_acts = defaultdict(int)
        for tag in self.dead_acts:
            new_dead_acts[f'dead_activations/{tag}'] = self.dead_acts[tag] / self.denoms[tag]
        self.dead_acts = new_dead_acts
            
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True
        
        
        
class RSVCallback:
    def __init__(self, group_size=200):
        if group_size <= 2 or group_size % 2:
            raise ValueError("group_size must be an even integer greater than two")
        self.group_size = group_size
        self.idx = 0
        self.is_able = False
        self.data = defaultdict(list)

    def __call__(self, module, input, output):
        if not self.is_able or not isinstance(
            module, (torch.nn.Linear, torch.nn.Conv2d)
        ):
            return
        if output.size(0) % self.group_size:
            raise ValueError("RSV batch size must be divisible by group_size")

        interval = self.group_size // 2
        for start in range(0, output.size(0), self.group_size):
            left_varying = output[start:start + interval]
            right_varying = output[start + interval:start + self.group_size]
            rsv = relative_source_variance(left_varying, right_varying)
            self.data[self.idx].append(rsv.detach().flatten())
        self.idx += 1

    def reset(self):
        self.data = defaultdict(list)
        self.idx = 0

    def gather_data(self):
        return self.data

    def disable(self):
        self.is_able = False

    def enable(self):
        self.is_able = True


CALLBACK_TYPE = {'dead_relu': DeadActivationCallback, 'rsv': RSVCallback}

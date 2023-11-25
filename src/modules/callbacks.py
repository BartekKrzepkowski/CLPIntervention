from collections import defaultdict

import torch
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
    def __init__(self):
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.idx = 0
        self.is_able = False
        self.data = defaultdict(list)
        
    def __call__(self, module, input, output):
        if (isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d)) and self.is_able: # czy moduł ma w sobie nazwę?
            for i in range(output.size(0) // self.group_size):
                interval = self.group_size // 2
                pack1 = output[i*self.group_size: i*self.group_size + interval]
                pack2 = output[i*self.group_size + interval: (i+1)*self.group_size]
                sv_a = pack1.var(dim=0)
                sv_b = pack2.var(dim=0)
                rsv = (sv_a - sv_b) / (sv_b + sv_a)
                rsv = rsv.view(-1)
                self.data[self.idx].append(rsv)
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
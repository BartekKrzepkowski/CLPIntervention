import torch

from src.modules.callbacks import CALLBACK_TYPE

class Hooks:
    def __init__(self, model, logger, callback_type):
        self.model = model
        self.logger = logger
        self.callback = CALLBACK_TYPE[callback_type]()
        self.hooks = []
        
    def register_hooks(self, model, modules_list):
        self.hooks = []
        for module in model.modules():
            if any(isinstance(module, module_type) for module_type in modules_list):
                self.hooks.append(module.register_forward_hook(self.callback))
                
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def write_to_tensorboard(self, step):
        self.callback.prepare_mean()
        self.logger.log_scalars(self.callback.dead_acts, step)
        self.reset()
        
    def write_to_tensorboard_histogram(self, step):
        self.callback.prepare_mean()
        self.logger.log_histogram(self.callback.data, step)
        self.reset()
            
    def reset(self):
        self.callback.reset()
        
    def disable(self):
        self.callback.disable()
        
    def enable(self):
        self.callback.enable()
        
    def gather_data(self):
        return self.callback.gather_data()
#!/usr/bin/env python3

from src.utils.prepare import prepare_model, prepare_loaders_clp


type_names = {
        'model': 'mm_mlp_bn',
        'criterion': 'cls',
        'dataset': 'dual_fmnist',
        'optim': 'sgd',
        'scheduler': 'multiplicative'
    }

dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': 0.0}
loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}

loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)


for i , data in enumerate(loaders['train']):
    pass

for i , data in enumerate(loaders['test_proper']):
    pass

for i , data in enumerate(loaders['test_blurred']):
    pass


from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler

GRAD_ACCUM_STEPS = 1
NUM_CLASSES = 200
RANDOM_SEED = 83
OVERLAP = 0.0


type_names = {
    'model': 'mm_effnetv2s',
    'criterion': 'cls',
    'dataset': 'dual_tinyimagenet',
    'optim': 'sgd',
    'scheduler': 'multiplicative'
}

dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': OVERLAP}
loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}

loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)

(x1, x2), y = next(iter(loaders['train']))

print(x1.shape, x2.shape, y.shape)
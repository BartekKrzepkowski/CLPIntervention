#!/usr/bin/env python3
import numpy as np
import torch

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_dual_clp import TrainerClassification
from src.trainer.trainer_context import TrainerContext


def objective(exp, window, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # model
    N = 1
    NUM_FEATURES = 3
    NUM_CLASSES = 10
    DIMS = [NUM_FEATURES, 32] + [64] * N + [128, NUM_CLASSES]
    CONV_PARAMS = {'img_height': 32, 'img_widht': 32, 'kernels': [3, 3] * (N + 1), 'strides': [1, 1] * (N + 1), 'paddings': [1, 1] * (N + 1), 'whether_pooling': [False, True] * (N + 1)}
    # trainer & schedule
    RANDOM_SEED = 83
    EPOCHS = epochs
    GRAD_ACCUM_STEPS = 1
    CLIP_VALUE = 100.0
    FP = 0.0#1e-2
    WD = 0.0
    LR = 2e-1
    OVERLAP = 0.0

    # prepare params
    type_names = {
        'model': 'dual_simple_cnn',
        'criterion': 'fp',
        'dataset': 'dual_cifar10',
        'optim': 'sgd',
        'scheduler': None
    }
    # wandb params
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_fp_{FP}_lr_{LR}_wd_{WD}'
    EXP_NAME = f'{GROUP_NAME}_window_{window} overlap={OVERLAP}, only left branch'
    PROJECT_NAME = 'Critical_Periods_Interventions'
    ENTITY_NAME = 'ideas_cv'

    h_params_overall = {
        'model': {'layers_dim': DIMS, 'activation_name': 'relu', 'conv_params': CONV_PARAMS},
        'criterion': {'model': None, 'general_criterion_name': 'ce', 'num_classes': NUM_CLASSES,
                      'whether_record_trace': False, 'fpw': FP},
        'dataset': {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True, 'overlap': OVERLAP},
        'loaders': {'batch_size': 125, 'pin_memory': True, 'num_workers': 8},
        'optim': {'lr': LR, 'momentum': 0.0, 'weight_decay': WD},
        'scheduler': None,
        'type_names': type_names
    }
    # set seed to reproduce the results in the future
    manual_seed(random_seed=RANDOM_SEED, device=device)
    # prepare model
    model = prepare_model(type_names['model'], model_params=h_params_overall['model']).to(device)
    # prepare criterion
    h_params_overall['criterion']['model'] = model
    criterion = prepare_criterion(type_names['criterion'], h_params_overall['criterion'])
    # prepare loaders
    loaders = prepare_loaders_clp(type_names['dataset'], h_params_overall['dataset'], h_params_overall['loaders'])
    # prepare optimizer & scheduler
    T_max = (len(loaders['train']) // GRAD_ACCUM_STEPS) * (window + epochs)
    print(T_max)
    # print(T_max//window, T_max-3*T_max//window, 3*T_max//window)
    # h_params_overall['scheduler'] = {'eta_max':LR, 'eta_medium':1e-2, 'eta_min':1e-6, 'warmup_iters2': 3*T_max//window, 'inter_warmups_iters': T_max-3*T_max//window, 'warmup_iters1': 3*T_max//window, 'milestones':[], 'gamma':1e-1}
    optim, lr_scheduler = prepare_optim_and_scheduler(model, type_names['optim'], h_params_overall['optim'],
                                                      type_names['scheduler'], h_params_overall['scheduler'])
    # DODAJ - POPRAWNE DANE
    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    # x_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x.pt').to(device)
    # y_data_proper = torch.load(f'data/{type_names["dataset"]}_held_out_proper_y.pt').to(device)
    
    # x_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x.pt').to(device)
    # y_data_blurred = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_y.pt').to(device) 
    
    # prepare trainer
    params_trainer = {
        'model': model,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'extra': None,#{'x_true1': x_data_proper, 'y_true1': y_data_proper, 'x_true2': x_data_blurred, 'y_true2': y_data_blurred, 'num_classes': NUM_CLASSES},
    }
    trainer = TrainerClassification(**params_trainer)

    # prepare run
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    config = TrainerContext(
        epoch_start_at=0,
        epoch_end_at=EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        save_multi=0,#T_max // 10,
        log_multi=1,#(T_max // EPOCHS) // 10,
        stiff_multi=(T_max // (window + epochs)) // 2,
        clip_value=CLIP_VALUE,
        base_path='reports',
        exp_name=EXP_NAME,
        logger_config={'logger_name': 'tensorboard', 'project_name': PROJECT_NAME, 'entity': ENTITY_NAME,
                       'hyperparameters': h_params_overall, 'whether_use_wandb': True,
                       'layout': ee_tensorboard_layout(params_names), 'mode': 'online'
                       },
        whether_disable_tqdm=True,
        random_seed=RANDOM_SEED,
        extra={'window': window, 'left_branch_interwention': None, 'right_branch_interwention': 'deactivation'},
        device=device
    )
    if exp == 'deficit':
        trainer.run_exp1(config)
    elif exp == 'sensitivity':
        trainer.run_exp2(config)
    elif exp == 'deficit_reverse':
        trainer.run_exp1_reverse(config)
    else:
        raise ValueError('exp should be either "deficit" or "sensitivity"')


if __name__ == "__main__":
    EPOCHS = 200
    for window in np.linspace(0, 200, 6):
        objective('deficit', int(window), EPOCHS)

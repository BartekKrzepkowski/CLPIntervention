from dataclasses import dataclass
from typing import Dict

import torch
import omegaconf


@dataclass
class TrainerContext:
    epoch_start_at: int = None,
    epoch_end_at: int = None,
    grad_accum_steps: int = None,
    save_multi: int = None,
    log_multi: int = None,
    stiff_multi: int = None,
    clip_value: float = None,
    base_path: str = None,
    exp_name: str = None,
    logger_config: Dict = None,
    whether_disable_tqdm: bool = None,
    random_seed: int = None,
    extra: Dict = None,
    device: torch.device = None

# config = TrainerContext(
#     epoch_start_at=0,
#     epoch_end_at=epochs,
#     grad_accum_steps=GRAD_ACCUM_STEPS,
#     save_multi=0,#T_max // 10,
#     log_multi=1,#(T_max // epochs) // 10,
#     stiff_multi=(T_max // (window + epochs)) // 2,
#     clip_value=CLIP_VALUE,
#     base_path='reports',
#     exp_name=EXP_NAME,
#     logger_config=logger_config,
#     whether_disable_tqdm=True,
#     random_seed=RANDOM_SEED,
#     extra=extra,
#     device=device
# )
import os
import warnings
import copy

import torch.nn
import peft
import torch.multiprocessing as mp
from easydict import EasyDict
from fling.utils import seed_everything
from fling.utils.config_utils import deep_merge_dicts, save_config_file

from zoo.default_config import default_exp_args


def compile_config(new_config: dict, seed: int) -> dict:
    r"""
    Overview:
        This function includes some important steps before the main process starts:
        1) Set the random seed for reproducibility.
        2) Determine the multiprocessing backend.
        3) Merge config (user config & default config).
        4) Compile data augmentation config.
        5) Create logging path and save the compiled config.
    Arguments:
        new_config: user-defined config.
        seed: random seed.
    Returns:
        result_config: the compiled config diction.
    """
    # Set random seed.
    seed_everything(seed)
    # Determine the multiprocessing backend.
    mp.set_start_method('spawn', force=True)

    merged_config = deep_merge_dicts(default_exp_args, new_config)
    result_config = EasyDict(merged_config)

    # Create logging path and save the compiled config.
    exp_dir = result_config.other.logging_path
    if not os.path.exists(exp_dir):
        try:
            os.makedirs(exp_dir)
        except FileExistsError:
            warnings.warn("Logging directory already exists.")
    save_config_file(result_config, os.path.join(exp_dir, 'total_config.py'))

    return result_config


def smart_copy(model: torch.nn.Module) -> torch.nn.Module:
    """
    Overview:
        If the input model is a peft model, deep-copy the attributes of peft_model except ``base_model``.
        Otherwise, deep-copy the complete model.
    Arguments:
        - model: The target model to be copied.
    Returns:
        - new_peft_model: The copied model.
    """
    # if isinstance(model, peft.PeftModel):
    #     new_dict = copy.copy(model.__dict__)
    #     for key, value in new_dict.items():
    #         if key != "base_model":
    #             new_dict[key] = copy.deepcopy(value)

    #     # for key in new_dict:
    #     #     print("Global Key:", key, type(new_dict[key]))

    #     for key in new_dict["_parameters"].keys():
    #         print("Global Key:", key, type(new_dict["_parameters"][key]))

    #     new_peft_model = copy.copy(model)
    #     new_peft_model.__dict__.update(new_dict)
    #     new_peft_model.base_model = model.base_model
    #     return new_peft_model
    # else:
    #     return copy.deepcopy(model)

    return model

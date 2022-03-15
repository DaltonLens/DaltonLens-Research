#!/usr/bin/env python3

import itertools
import subprocess
from tqdm import tqdm
import sys
import shutil
from typing import Dict, List
from types import SimpleNamespace

from dlcharts.common.utils import printBold
from dlcharts.common.timer import Timer

def quick_validate():
    subprocess.run([
            "python3", "scripts/train_regression.py",
            f"quick",
            
            "--clean_previous",

            "--batch_size", "4",

            "--validate",

            "--model", "uresnet18-v1-residual"
        ], check=True)

def cartesian_product (params: Dict[str,List]):
    """Generate a list of dictionaries from a dictionary of list which
    corresponds to the cartesian product of the input values.

    Example:
    
    Input: [Dict[str,List]]
    { 
        'param1': [value1, value2],
        'param2': [value3, value4],
    }

    Output: List[Dict[str,str]]
    [
        {'param1': value1, 'param2': value3 },
        {'param1': value1, 'param2': value4 },
        {'param1': value2, 'param2': value3 },
        {'param1': value2, 'param2': value4 },
    ]
    """
    values_product = itertools.product(*params.values())
    keys = params.keys()
    params_list = [dict(zip(keys, values)) for values in values_product]
    return params_list

if __name__ == "__main__":

    # quick_validate()
    # sys.exit(0)

    # params_set = dict(
    #     model=["uresnet18-v1", "uresnet18-v1-residual"],
    #     decoder_lr=["5e-3", "1e-3", "1e-4"],
    #     batch_size = ["32", "64", "128"]
    #     loss = ["mse", "l1"]
    # )

    params_set = dict(
        model=["uresnet18-v1-residual"],
        decoder_lr=["5e-3"],
        batch_size = ["32"],
        # loss = ["l1"]
        loss = ["mse"]
    )
   
    # [dict1, dict2, ...]
    params_dicts = cartesian_product (params_set)

    for idx, p in enumerate(params_dicts):
        printBold (f"[{idx}/{len(params_dicts)}] training {p}")
        p = SimpleNamespace(**p)
        with Timer("Train one config"):
            subprocess.run([
                "python3", "scripts/train_regression.py",
                # v4 is the input data / data augmentation version.
                f"v4_{p.model}_{p.loss}_bn{p.batch_size}_{p.decoder_lr}",
                
                # "--clean_previous",
                
                "--model", p.model,
                "--batch_size", p.batch_size,
                "--decoder_lr", p.decoder_lr,
                "--encoder_lr", "1e-5",
                "--loss", p.loss,

                "--epochs_decoder_only", "10",
                "--epochs_finetune", "200",

                # TEMP!
                # "--debug",
                # "--validate",
                # "--overfit", "1",
                # "--batch_size", "4",
            ], check=True)

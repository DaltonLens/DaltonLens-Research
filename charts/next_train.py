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
            "python3", "scripts/train_regression_gated.py",
            f"quick",

            "--loss", "mse",

            "--clean_previous",

            "--batch_size", "4",

            "--validate",

            "--overfit", "4",

            # "--debug",

            "--epochs_decoder_only", "100",

            "--epochs_finetune", "1",

            "--no-evaluation",

            "--model", "unet-rn18-rn18",
            # "--model", "unet-mobv2-rn18",
            # "--model", "unet-mobv2-rn18-nopretrain",
            # "--model", "unet-mobv2-mobv2",
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

    quick_validate()
    sys.exit(0)

    # params_set = dict(
    #     model=["uresnet18", "uresnet18-sa"],
    #     decoder_lr=["5e-3", "1e-3", "1e-4"],
    #     batch_size = ["32", "64", "128"]
    #     loss = ["mse", "l1"]
    # )

    params_set = dict(
        # model=["uresnet18-sa", "uresnet18-no-residual", "uresnet18", "uresnet18-shuffle", "uresnet18-sa-shuffle"],
        model=["uresnet18"],
        # model=["unet-mobv2-large", "unet-mobv2-rn18", "unet-mobv2-medium"],
        # model=["uresnet18-sa-shuffle"],
        encoder_lr=["1e-5"],
        decoder_lr=["5e-3"],
        batch_size = ["32"],
        # loss = ["l1"],
        # loss = ["mse_and_fg_var"],
        loss = ["mse"],

        epochs = [(20, 100)],
    )
   
    # [dict1, dict2, ...]
    params_dicts = cartesian_product (params_set)

    # Add a config without pretraining.
    # params_dicts.insert(0, dict(
    #     model="unet-mobv2-rn18-nopretrain",
    #     encoder_lr="1e-4",
    #     decoder_lr="5e-3",
    #     batch_size = "32",
    #     loss = "mse",
    #     epochs = (1, 200),
    # ))

    for idx, p in enumerate(params_dicts):
        printBold (f"[{idx}/{len(params_dicts)}] training {p}")
        p = SimpleNamespace(**p)
        with Timer("Train one config"):
            subprocess.run([
                "python3", "scripts/train_regression_gated.py",
                # v4 is the input data / data augmentation version.
                # the 'g' stands for gated regression
                f"v4_gated_{p.model}_{p.loss}_bn{p.batch_size}_{p.decoder_lr}_{p.encoder_lr}",
                               
                "--model", p.model,
                "--batch_size", p.batch_size,
                "--decoder_lr", p.decoder_lr,
                "--encoder_lr", p.encoder_lr,
                "--loss", p.loss,

                "--epochs_decoder_only", str(p.epochs[0]),
                "--epochs_finetune", str(p.epochs[1]),

                # TEMP!
                "--debug",
                "--clean_previous",
                "--validate",
                "--overfit", "1",
                "--no-evaluation",
                "--batch_size", "4",
            ], check=True)

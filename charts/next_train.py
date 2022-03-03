#!/usr/bin/env python3

import subprocess
from tqdm import tqdm
import sys
import shutil

def quick_validate():
    subprocess.run([
            "python3", "scripts/train_regression.py",
            f"quick",
            
            "--clean_previous",

            "--batch_size", "4",

            "--validate",

            "--model", "uresnet18-v1-residual"
        ], check=True)

if __name__ == "__main__":

    quick_validate()
    sys.exit(0)

    version = 0
    for model in ["uresnet18-v1", "uresnet18-v1-residual"]:
        for decoder_lr in ["5e-3", "1e-3", "1e-4"]:
            for batch_size in ["32", "64", "128"]:
                subprocess.run([
                    "python3", "scripts/train_regression.py",
                    f"{model}_bn{batch_size}_{decoder_lr}",
                    
                    "--batch_size", batch_size,
                    "--decoder_lr", decoder_lr,
                    "--encoder_lr", "1e-5",
                    "--epochs_decoder_only", "30",
                    "--epochs_finetune", "10",

                    "--model", model,

                    # TEMP!
                    # "--validate",
                    # "--overfit", "1",
                    # "--batch_size", "4",
                ], check=True)
                version += 1

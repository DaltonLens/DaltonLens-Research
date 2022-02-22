#!/usr/bin/env python3

import subprocess
from tqdm import tqdm

if __name__ == "__main__":
    version = 0
    for decoder_lr in tqdm([5e-3, 1e-3, 1e-4]):
        subprocess.run([
            "python3", "scripts/train_regression.py",
            f"baseline-v{version:02d}",
            
            "--batch_size", "64",
            "--decoder_lr", str(decoder_lr),
            "--encoder_lr", "1e-5",
            "--epochs_decoder_only", "30",
            "--epochs_finetune", "10",            

            # TEMP!
            # "--validate",
            # "--overfit", "1",
            # "--batch_size", "4",
        ], check=True)
        version += 1

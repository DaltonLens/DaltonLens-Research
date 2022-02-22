#!/usr/bin/env python3

import shutil
import dlcharts
from torchmetrics import Accuracy
import dlcharts.pytorch.color_regression as cr
from dlcharts.pytorch.utils import is_google_colab, num_trainable_parameters, debugger_is_active, evaluating
from dlcharts.pytorch.lightning import GlobalProgressBar, ValidationStepCallback, ColabCheckpointIO
from dlcharts.common.utils import printBold

import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import torch.utils.tensorboard

import numpy as np

from icecream import ic

import argparse
from typing import Optional
from pathlib import Path
from enum import Enum
import logging
import math
import warnings
import os
import sys

DEFAULT_BATCH_SIZE=64 if is_google_colab() else 4
WORKERS=0 if debugger_is_active() else os.cpu_count()

class DrawingsDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: Path, batch_size:int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_path = dataset_path
        self.preprocessor = cr.ImagePreprocessor(None, target_size=128)
        self.dataset = cr.ColorRegressionImageDataset(dataset_path, self.preprocessor)
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        n_train = max(int(len(self.dataset) * 0.7), 1)
        n_val = len(self.dataset) - n_train
        generator = torch.Generator().manual_seed(42)
        np_gen = np.random.default_rng(42)

        train_indices = np.array(range(0, n_train))
        np_gen.shuffle(train_indices)

        val_indices = list(range(n_train, len(self.dataset)))
        np_gen.shuffle(val_indices)

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)

        # We need to preload it to know its size and configure the scheduler.
        self.preloaded_train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=WORKERS)

        self.val_dataloader_for_training_step = DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=WORKERS)

        # They are shuffled, so we're fine.
        self.monitored_train_samples = train_indices[0:5]
        self.monitored_val_samples = val_indices[0:5]

    def train_dataloader(self):
        return self.preloaded_train_dataloader

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=WORKERS)

class RegressionValidationStepCallback(ValidationStepCallback):
    def __init__(self, datamodule: pl.LightningDataModule):
        super().__init__(datamodule)

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        def evaluate_images_at_indices(indices):
            inputs = []
            outputs = []
            targets = []
            for idx in indices:
                input,target,_ = pl_module.transfer_batch_to_device(self.datamodule.dataset[idx], pl_module.device, 0)
                output = pl_module(input.unsqueeze(0)).squeeze(0)
                inputs.append(trainer.datamodule.preprocessor.denormalize_and_clip_as_tensor(input.detach().cpu()))
                outputs.append(trainer.datamodule.preprocessor.denormalize_and_clip_as_tensor(output.detach().cpu()))
                targets.append(trainer.datamodule.preprocessor.denormalize_and_clip_as_tensor(target.detach().cpu()))
            return torch.cat([torch.cat(outputs, dim=2), torch.cat(targets, dim=2), torch.cat(inputs, dim=2)], dim=1)
        
        with evaluating(pl_module):
            results_train = evaluate_images_at_indices(trainer.datamodule.monitored_train_samples)
            pl_module.writer.add_image("Train Samples", results_train, trainer.current_epoch)

            results_val = evaluate_images_at_indices(trainer.datamodule.monitored_val_samples)
            pl_module.writer.add_image("Val Samples", results_val, trainer.current_epoch)

        return super().on_epoch_end(trainer, pl_module)

class Phase(Enum):
    DecoderOnly = 0
    FineTune = 1

def regression_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    diff = torch.abs(outputs-labels)
    max_diff = torch.max(diff, dim=1)[0]
    num_good = torch.count_nonzero(max_diff < ((20/255.0)*2.0))
    num_pixels = max_diff.numel()
    accuracy = num_good / num_pixels
    return accuracy

class RegressionModule(pl.LightningModule):
    def __init__(self, phase: Phase, encoder_lr=1e-4, decoder_lr=1e-3, regression_model: str = 'uresnet18-v1'):
        super().__init__()
        self.save_hyperparameters()
        self.model = dlcharts.pytorch.models.create_regression_model(regression_model)
        
        # Aliases to get nice summary stats.
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

        self.loss_fn = nn.MSELoss()
        self.val_iterator_per_training_step = None

        self.accuracy_fn = regression_accuracy

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.model (x)
        return y

    def on_fit_start(self) -> None:
        if self.hparams.phase == Phase.DecoderOnly:
            self.model.freeze_encoder ()
        else:
            self.model.unfreeze_encoder ()
        ic(num_trainable_parameters(self.model))
        return super().on_fit_start()

    def training_step(self, batch, batch_idx):
        inputs, labels, json_files = batch
        outputs = self(inputs)                        
        loss = self.loss_fn(outputs, labels)
        accuracy = self.accuracy_fn(outputs, labels)
        
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.writer.add_scalars('loss_iter', dict(train=loss), self.global_step)
            self.writer.add_scalars('accuracy_iter', dict(train=accuracy), self.global_step)            

        self.log('acc', accuracy, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        return dict(loss=loss, accuracy=accuracy)

    def validation_step(self, batch, batch_idx):
        inputs, labels, json_files = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        accuracy = self.accuracy_fn(outputs, labels)
        return dict(loss=loss, accuracy=accuracy)

    @property
    def writer(self) -> torch.utils.tensorboard.writer.SummaryWriter:
        return self.logger.experiment

    def training_epoch_end(self, outputs) -> None:
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.writer.add_scalars('loss_epoch', dict(train=loss), self.current_epoch)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack([o['loss'] for o in outputs]).mean()
        accuracy = torch.stack([o['accuracy'] for o in outputs]).mean()
        self.writer.add_scalars('loss_epoch', dict(val=loss), self.current_epoch)
        self.writer.add_scalars('accuracy_epoch', dict(val=accuracy), self.current_epoch)
        self.log("hp_metric", accuracy)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.hparams.encoder_lr },
            {'params': self.decoder.parameters(), 'lr': self.hparams.decoder_lr }
        ])

        dm = self.trainer.datamodule
        train_loader_len = len(dm.preloaded_train_dataloader)
        assert self.trainer.accumulate_grad_batches == 1 # not supported yet, would need to check the math
        steps_per_epoch = train_loader_len
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, [self.hparams.encoder_lr, self.hparams.decoder_lr], steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }


def fit_decoder_only(args, dataset_path: Path, xp_dir, trainer_common_params):
    data = DrawingsDataModule(dataset_path, batch_size=args.batch_size)

    tb_logger = TensorBoardLogger(xp_dir/'decoder-only', name='', version='tb')

    trainer = pl.Trainer(
        max_epochs=args.epochs_decoder_only,
        
        default_root_dir=xp_dir / 'decoder-only',
        
        callbacks=[
            ModelCheckpoint(dirpath=xp_dir/'decoder-only', save_last=True),
            RegressionValidationStepCallback(data),
            GlobalProgressBar(),
        ],

        logger=tb_logger,

        **trainer_common_params
    )

    ckpt_path = xp_dir/'decoder-only'/'last.ckpt'
    if not ckpt_path.exists():
        ckpt_path = None    

    model = RegressionModule(
        phase=Phase.DecoderOnly, 
        regression_model=args.model, 
        encoder_lr=0.0, 
        decoder_lr=args.decoder_lr
    )
    trainer.fit(model, data, ckpt_path=ckpt_path)
    
def finetune(args, dataset_path: Path, xp_dir, trainer_common_params):
    data = DrawingsDataModule(dataset_path, batch_size=args.batch_size)

    ckpt_path = xp_dir/'finetune'/'last.ckpt'
    if not ckpt_path.exists():
        ckpt_path = None

    tb_logger = TensorBoardLogger(xp_dir/'finetune', name='', version='tb')

    trainer = pl.Trainer(
        max_epochs=args.epochs_finetune,

        callbacks=[
            ModelCheckpoint(dirpath=xp_dir/'finetune', save_last=True),
            RegressionValidationStepCallback(data),
            GlobalProgressBar(),
        ],

        logger=tb_logger,

        **trainer_common_params
    )

    model = RegressionModule.load_from_checkpoint(
        checkpoint_path=xp_dir/'decoder-only'/'last.ckpt', 
        phase=Phase.FineTune,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        regression_model=args.model
    )

    trainer.fit(model, data, ckpt_path=ckpt_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--clean_previous", action='store_true')
    
    parser.add_argument("--epochs_decoder_only", type=int, default=40)
    parser.add_argument("--epochs_finetune", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--model", type=str, default="uresnet18-v1")
    parser.add_argument("--decoder_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)

    
    args = parser.parse_args()
    if args.validate:
        args.epochs_decoder_only = 2
        args.epochs_finetune = 2

    return args

if __name__ == "__main__":
    args = parse_args()

    if False and not args.validate:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.simplefilter("ignore", LightningDeprecationWarning)
        warnings.filterwarnings("ignore", '.*')

    root_dir = Path(__file__).parent.parent
    dataset_path = Path("/content/datasets/drawings") if is_google_colab() else root_dir / 'inputs' / 'opencv-generated' / 'drawings'

    trainer_common_params = dict(
        # gpus=1,

        plugins = [ColabCheckpointIO()],

        overfit_batches = args.overfit if args.overfit != 0 else 0.0,

        # fast_dev_run = True

        val_check_interval=1.0,
        log_every_n_steps=5,

        limit_val_batches=1 if args.validate else 1.0,
        limit_train_batches=1 if args.validate else 1.0,

        # profiler="simple",
    )

    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2006
    #   Discusses various options
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3095
    #   Option calling fit twice

    xp_name = args.name
    xp_dir = root_dir / 'logs' / xp_name

    if args.clean_previous:
        print(f"Warning: cleaning {xp_dir}")
        shutil.rmtree(xp_dir, ignore_errors=True)

    print (f"Training experiment stored in {xp_dir}")
    print()

    printBold (">>> Training the decoder only...")
    fit_decoder_only (args, dataset_path, xp_dir, trainer_common_params)

    dlcharts.pytorch.utils.clear_gpu_memory()

    print()
    printBold (">>> Fine-tuning the entire model...")
    finetune (args, dataset_path, xp_dir, trainer_common_params)

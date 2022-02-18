#!/usr/bin/env python3

import dlcharts
import dlcharts.pytorch.color_regression as cr
from dlcharts.pytorch.utils import is_google_colab
from dlcharts.common.utils import InfiniteIterator

import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import torch.utils.tensorboard

from typing import Optional
from pathlib import Path
import os

DEFAULT_BATCH_SIZE=64 if is_google_colab() else 4
WORKERS=os.cpu_count() if is_google_colab() else 0

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

        train_indices = range(0, n_train)
        val_indices = range(n_train, len(self.dataset))

        self.train_sampler = SubsetRandomSampler(train_indices, generator=generator)
        self.val_sampler = SubsetRandomSampler(val_indices, generator=generator)

        # We need to preload it to know its size and configure the scheduler.
        self.preloaded_train_dataloader = DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, num_workers=WORKERS)

        self.val_sampler_for_training_step = SubsetRandomSampler(val_indices, generator=generator)
        self.val_dataloader_for_training_step = DataLoader(self.dataset, sampler=self.val_sampler_for_training_step, batch_size=self.batch_size, num_workers=WORKERS)

    def train_dataloader(self):
        return self.preloaded_train_dataloader

    def val_dataloader(self):
        return DataLoader(self.dataset, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=WORKERS)


class ValidationStepCallback(pl.Callback):
    def __init__(self, datamodule: pl.LightningDataModule):
        self.datamodule = datamodule

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.infinite_iterator = InfiniteIterator(self.datamodule.val_dataloader_for_training_step)
        return super().on_fit_start(trainer, pl_module)

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.global_step % trainer.log_every_n_steps == 0:
            batch = next(self.infinite_iterator)
            batch_device = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            
            with torch.no_grad():
                pl_module.eval()
                val_loss = pl_module.validation_step(batch_device, 0)
                pl_module.train()
            
            pl_module.writer.add_scalars('loss_iter', dict(val=val_loss), pl_module.global_step)
        return super().on_batch_end(trainer, pl_module)

class RegressionModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.model = dlcharts.pytorch.models.create_regression_model('uresnet18-v1')
        self.loss_fn = nn.MSELoss()
        self.val_iterator_per_training_step = None

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = self.model (x)
        return y

    def on_fit_start(self) -> None:
        return super().on_fit_start()

    def training_step(self, batch, batch_idx):
        inputs, labels, json_files = batch
        outputs = self(inputs)                
        
        loss = self.loss_fn(outputs, labels)
        
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.writer.add_scalars('loss_iter', dict(train=loss), self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, json_files = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss

    @property
    def writer(self) -> torch.utils.tensorboard.writer.SummaryWriter:
        return self.logger.experiment

    def training_epoch_end(self, outputs) -> None:
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.writer.add_scalars('loss_epoch', dict(train=loss), self.current_epoch)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs) -> None:
        loss = torch.stack(outputs).mean()
        self.writer.add_scalars('loss_epoch', dict(val=loss), self.current_epoch)
        self.log("hp_metric", loss)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        dm = self.trainer.datamodule
        train_loader_len = len(dm.preloaded_train_dataloader)
        steps_per_epoch = (train_loader_len//dm.hparams.batch_size)//self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.hparams.lr, steps_per_epoch=steps_per_epoch, epochs=self.trainer.max_epochs)
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                },
            }

if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent
    dataset_path = Path("/content/datasets/drawings") if is_google_colab() else root_dir / 'inputs' / 'opencv-generated' / 'drawings'
    data = DrawingsDataModule(dataset_path)
    model = RegressionModule()
    trainer = pl.Trainer(
        # gpus=1,
        max_epochs=5, 
        val_check_interval=1.0, 
        limit_val_batches=2, 
        limit_train_batches=10,
        log_every_n_steps=1,
        # fast_dev_run = 1,
        callbacks=[ValidationStepCallback(data)])

    trainer.fit(model, data)

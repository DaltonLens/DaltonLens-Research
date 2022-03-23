#!/usr/bin/env python3

import shutil
import dlcharts
from torchmetrics import Accuracy
import dlcharts.pytorch.color_regression as cr
from dlcharts.pytorch.utils import is_google_colab, num_trainable_parameters, debugger_is_active, evaluating, Experiment
from dlcharts.common.utils import InfiniteIterator, printBold, swap_rb

import torch
from torch.nn import functional as F
from torch import nn

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import torch.utils.tensorboard

import numpy as np
import cv2

from icecream import ic

import argparse
from typing import Optional, List
from pathlib import Path
from enum import Enum
import logging
import math
import warnings
import os
import sys
from dataclasses import dataclass

from tqdm import tqdm

from zv.log import zvlog

Sample = cr.ColorRegressionImageDataset.Sample

DEFAULT_BATCH_SIZE=64 if is_google_colab() else 4
WORKERS=0 if debugger_is_active() else os.cpu_count()
@dataclass
class Hparams:
    batch_size: int = 4
    encoder_lr: float = 1e-5
    decoder_lr: float = 1e-3
    regression_model: str = 'invalid'
    loss: str = 'mse'

@dataclass
class Params:
    name: str
    logs_dir: Path
    device: torch.device
    overfit: int = 0
    clear_previous_results: bool = False
    clear_top_folder: bool = False
    num_frozen_epochs: int = 10
    num_finetune_epochs: int = 5    

    @property
    def num_epochs(self):
        return self.num_frozen_epochs + self.num_finetune_epochs
        
@dataclass
class EpochMetrics:
    training_loss: float
    val_loss: float
    val_fg_accuracy: float
    val_bg_accuracy: float
    
class DrawingsData:
    train_dataloader: DataLoader
    val_dataloader: DataLoader

    def __init__(self, dataset_path: List[Path], params: Params, hparams: Hparams):
        super().__init__()
        self.params = params
        self.hparams = hparams
        self.preprocessor = cr.ImagePreprocessor(None, target_size=192)
        datasets = [cr.ColorRegressionImageDataset(path, self.preprocessor) for path in dataset_path]
        self.dataset = torch.utils.data.ConcatDataset(datasets)
        
        self.generator = torch.Generator().manual_seed(42)
        self.np_gen = np.random.default_rng(42)

        if params.overfit != 0:
            indices = self.np_gen.choice(range(0, len(self.dataset)), size=params.overfit + 1, replace=False)
            self.dataset = torch.utils.data.Subset(self.dataset, indices)
        self.create ()

    def create(self):
        if params.overfit != 0:
            n_train = params.overfit
            n_val = len(self.dataset) - n_train
        else:
            n_train = max(int(len(self.dataset) * 0.7), 1)
            n_val = len(self.dataset) - n_train

        all_indices = np.array(range(0, len(self.dataset)))
        self.np_gen.shuffle(all_indices)

        train_indices = all_indices[0:n_train]
        val_indices = all_indices[n_train:]

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.dataset, val_indices)

        self.train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size, num_workers=WORKERS)
        self.val_dataloader = DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=WORKERS)

        self.val_dataloader_for_training_step = DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size, num_workers=WORKERS)

        # They are shuffled, so we're fine.
        self.monitored_train_samples = [self.train_dataset[idx] for idx in range(0,min(3, n_train))]
        self.monitored_val_samples = [self.val_dataset[idx] for idx in range(0,min(10, n_val))]

def regression_accuracy(outputs: torch.FloatTensor, batch: Sample):
    diff = torch.abs(outputs-batch.labels_rgb)
    max_diff = torch.max(diff, dim=1)[0]
    
    # Very important to call numel on the image with one channel,
    # otherwise it'll be x3.
    num_pixels = max_diff.numel()

    good_values = (max_diff < (20/255.0))
    fg_mask = batch.labels_mask > 0
    num_fg_pixels = torch.count_nonzero(fg_mask)
    num_bg_pixels = num_pixels - num_fg_pixels
    fg_good = torch.logical_and (good_values, fg_mask)
    acc_fg = torch.count_nonzero(fg_good) / num_fg_pixels

    bg_good = torch.logical_and (good_values, torch.logical_not(fg_mask))
    acc_bg = torch.count_nonzero(bg_good) / num_bg_pixels
    return torch.tensor([acc_fg, acc_bg])

def weighted_loss(loss_fn):
    def loss(outputs: torch.FloatTensor, batch: Sample):
        raw_loss = loss_fn(outputs, batch.labels_rgb)
        # 1 for bg, 10 for foreground
        # Typical ratio in our training images is more like 2x
        # more background. But does not hurt to make it even stronger
        # for the foreground, since this is the hard part.
        weights = (batch.labels_mask > 0).float() * 10.0 + 1.0
        weighted_loss = raw_loss * weights.unsqueeze(1) # add the channel dimension to broadcast
        loss_scalar = torch.mean (weighted_loss)
        return loss_scalar
    return loss

def uniform_loss(loss_fn):
    def loss (outputs: torch.FloatTensor, batch: Sample):
        return loss_fn(outputs, batch.labels_rgb)
    return loss

class RegressionTrainer:
    def __init__(self, params: Params, hparams: Hparams):
        self.hparams = hparams
        self.params = params
        self.model = dlcharts.pytorch.models.create_regression_model(hparams.regression_model)            

        self.device = self.params.device

        logs_root_dir = dlcharts.pytorch.utils.default_output_dir
        self.xp = Experiment(params.name + ('-overfit' if params.overfit else ''),
                             params.logs_dir,
                             clear_previous_results=params.clear_previous_results,
                             clear_top_folder=params.clear_top_folder)

        losses = dict(
            mse = uniform_loss(nn.MSELoss()),
            l1 = uniform_loss(nn.L1Loss()),
            weighted_mse = weighted_loss(nn.MSELoss(reduction='none')),
        )
        self.loss_fn = losses[hparams.loss]
        self.accuracy_fn = regression_accuracy

    def train(self, data: DrawingsData):
        self.model.to(self.device)
        
        self.data = data
        self.optimizer = self._create_optimizer()
        frozen_scheduler = self._create_scheduler(data, frozen=True)
        finetune_scheduler = self._create_scheduler(data, frozen=False)

        sample_input = data.dataset[0].labels_rgb.unsqueeze(0).to(self.device)
        schedulers = dict(frozen_scheduler=frozen_scheduler,finetune_scheduler=finetune_scheduler)
        self.xp.prepare ("default", self.model, self.optimizer, schedulers, self.device, sample_input)

        self.model.freeze_encoder()
        print (f"Encoder frozen: {num_trainable_parameters (self.model)} params.")
        self.current_scheduler = frozen_scheduler
        pbar = tqdm(range(self.xp.first_epoch, self.params.num_epochs))
        for e in range(self.xp.first_epoch, self.params.num_frozen_epochs):
            metrics = self._train_one_epoch (e)
            pbar.set_postfix({
                'mode': 'decoder-only', 
                'train_loss': metrics.training_loss, 
                'val_loss': metrics.val_loss, 
                'val_acc_fg': metrics.val_fg_accuracy,
                'val_acc_bg': metrics.val_bg_accuracy
            })
            pbar.update()

        self.model.unfreeze_encoder()
        print (f"Encoder trainable: {num_trainable_parameters (self.model)} params.")
        self.current_scheduler = finetune_scheduler
        for e in range(max(self.xp.first_epoch, self.params.num_frozen_epochs), self.params.num_epochs):
            metrics = self._train_one_epoch (e)
            pbar.set_postfix({
                'mode': 'finetune', 
                'train_loss': metrics.training_loss, 
                'val_loss': metrics.val_loss, 
                'val_acc_fg': metrics.val_fg_accuracy, 
                'val_acc_bg': metrics.val_bg_accuracy})
            pbar.update()

        self.xp.finalize (vars(self.hparams), dict(acc_fg=metrics.val_fg_accuracy, acc_bg=metrics.val_bg_accuracy))

    def _train_one_epoch(self, current_epoch) -> EpochMetrics:
        self.current_epoch = current_epoch
        self.model.train()
        num_batches = len(self.data.train_dataloader)

        cumulated_train_loss = torch.tensor(0.0)

        val_batch_iterator = InfiniteIterator(self.data.val_dataloader)

        pbar = tqdm(self.data.train_dataloader, position=1, leave=False)
        batch: Sample
        for batch_idx, batch in enumerate(pbar):
            self.global_step = current_epoch * num_batches + batch_idx            
            batch = batch.to(self.device)
            outputs = self._evaluate_batch (batch)
            loss = self.loss_fn(outputs, batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.current_scheduler.step()
            cumulated_train_loss += loss.cpu()

            if batch_idx % 10 == 0:
                with torch.no_grad():
                    fg_bg_accuracy = self.accuracy_fn(outputs, batch)
                self.xp.writer.add_scalars('loss_iter', dict(train=loss), self.global_step)
                self.xp.writer.add_scalars('accuracy_iter_fg', dict(train=fg_bg_accuracy[0]), self.global_step)
                self.xp.writer.add_scalars('accuracy_iter_bg', dict(train=fg_bg_accuracy[1]), self.global_step)

            if batch_idx % 20 == 0:
                self._compute_batch_validation(val_batch_iterator)

            if current_epoch == 0 and batch_idx % 10 == 0:
                self._compute_monitored_images()

        train_loss_epoch = cumulated_train_loss / num_batches
        self.xp.writer.add_scalars('loss_epoch', dict(train=train_loss_epoch), current_epoch)

        epoch_val_loss, epoch_val_fg_bg_accuracy = self._compute_epoch_validation()
        metrics = EpochMetrics(training_loss=train_loss_epoch.item(),
                               val_loss=epoch_val_loss.item(),
                               val_fg_accuracy=epoch_val_fg_bg_accuracy[0].item(),
                               val_bg_accuracy=epoch_val_fg_bg_accuracy[1].item())

        self._compute_monitored_images()

        if (self.current_epoch > 0
            and self.current_epoch % 10 == 0
            or self.current_epoch == self.params.num_frozen_epochs-1
            or self.current_epoch == self.params.num_epochs-1):
            self.xp.save_checkpoint(self.current_epoch)

        return metrics
            
    def _compute_monitored_images(self):
        def log_and_save(name, im):
            im = (im*255.9999).astype(np.uint8)
            zvlog.image(name, im)
            # opencv expects bgr
            cv2.imwrite(str(self.xp.log_path / (name + '.png')), swap_rb(im))
            
        def evaluate_images(title, sample_list):
            inputs = []
            outputs = []
            targets = []
            sample: Sample
            for idx, sample in enumerate(sample_list):
                sample_device = sample.to(self.device)
                output = self._evaluate_single_item (sample_device)
                inputs.append(self.data.preprocessor.denormalize_and_clip_as_tensor(sample_device.image.detach().cpu()))
                outputs.append(self.data.preprocessor.denormalize_and_clip_as_tensor(output.detach().cpu()))
                targets.append(self.data.preprocessor.denormalize_and_clip_as_tensor(sample_device.labels_rgb.detach().cpu()))
            epoch = self.current_epoch
            for idx in range(0, len(outputs)):                
                log_and_save (f"{title}-{idx}-epoch{epoch}-input", inputs[idx].permute(1, 2, 0).numpy())
                log_and_save (f"{title}-{idx}-epoch{epoch}-output", outputs[idx].permute(1, 2, 0).numpy())
                log_and_save (f"{title}-{idx}-epoch{epoch}-target", targets[idx].permute(1, 2, 0).numpy())
            stacked_for_tboard = torch.cat([torch.cat(outputs, dim=2), torch.cat(targets, dim=2), torch.cat(inputs, dim=2)], dim=1)
            self.xp.writer.add_image(title, stacked_for_tboard, epoch)
        
        with evaluating(self.model), torch.no_grad():
            evaluate_images("Train Samples", self.data.monitored_train_samples)
            evaluate_images("Val Samples", self.data.monitored_val_samples)

    def _compute_batch_validation(self, val_batch_iterator):
        with evaluating(self.model), torch.no_grad():
            batch: Sample = next(val_batch_iterator).to(self.device)
            outputs = self._evaluate_batch (batch)
            val_loss = self.loss_fn(outputs, batch)
            val_fg_bg_accuracy = self.accuracy_fn(outputs, batch)
            self.xp.writer.add_scalars('loss_iter', dict(val=val_loss), self.global_step)
            self.xp.writer.add_scalars('accuracy_iter_fg', dict(val=val_fg_bg_accuracy[0]), self.global_step)
            self.xp.writer.add_scalars('accuracy_iter_bg', dict(val=val_fg_bg_accuracy[1]), self.global_step)

    def _evaluate_batch(self, batch: Sample):
        outputs = self.model(batch.image.to(self.device))
        return outputs

    def _evaluate_single_item(self, item):
        outputs = self.model(item.image.to(self.device).unsqueeze(0)).squeeze(0)
        return outputs

    def _compute_epoch_validation(self):
        with evaluating(self.model), torch.no_grad():
            cumulated_val_loss = torch.tensor(0.)
            cumulated_val_fg_bg_accuracy = torch.tensor([0.,0.])
            num_batches = len(self.data.val_dataloader)
            batch: Sample
            for batch in self.data.val_dataloader:
                batch = batch.to(self.device)
                outputs = self._evaluate_batch (batch)
                cumulated_val_loss += self.loss_fn(outputs, batch).cpu()
                cumulated_val_fg_bg_accuracy += self.accuracy_fn(outputs, batch).cpu()
            val_loss = cumulated_val_loss / num_batches
            val_fg_bg_accuracy = cumulated_val_fg_bg_accuracy / num_batches
            
            self.xp.writer.add_scalars('loss_epoch', dict(val=val_loss), self.current_epoch)
            self.xp.writer.add_scalars('accuracy_epoch_fg', dict(val=val_fg_bg_accuracy[0]), self.current_epoch)
            self.xp.writer.add_scalars('accuracy_epoch_bg', dict(val=val_fg_bg_accuracy[1]), self.current_epoch)
            return val_loss, val_fg_bg_accuracy

    def _create_scheduler(self, data: DrawingsData, frozen: bool):
        lr = [self.hparams.encoder_lr, self.hparams.decoder_lr]
        steps_per_epoch = len(data.train_dataloader)
        epochs = self.params.num_frozen_epochs if frozen else self.params.num_finetune_epochs
        return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    def _create_optimizer(self):
        # LR will get replaced after by the scheduler.
        return torch.optim.AdamW([
            {'params': self.model.encoder.parameters(), 'lr': 0 },
            {'params': self.model.decoder.parameters(), 'lr': 0 }
        ])

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument("--debug", action='store_true', help='Enable zvlog.')
    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--clean_previous", action='store_true')
    
    parser.add_argument("--epochs_decoder_only", type=int, default=40)
    parser.add_argument("--epochs_finetune", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--model", type=str, default="uresnet18")
    parser.add_argument("--decoder_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="mse")
    
    args = parser.parse_args()
    if args.validate:
        args.overfit = 64*10
        args.epochs_decoder_only = 3
        args.epochs_finetune = 3
    return args

if __name__ == "__main__":
    args = parse_command_line()
    if args.debug:
        zvlog.start ()
        # zvlog.start (('127.0.0.1', 7007))

    root_dir = Path(__file__).parent.parent
    datasets_path = [
        root_dir / 'inputs' / 'train' / 'opencv-generated' / 'drawings',
        root_dir / 'inputs' / 'train' / 'opencv-generated-background',
        root_dir / 'inputs' / 'train' / 'mpl-generated',
        root_dir / 'inputs' / 'train' / 'mpl-generated-no-antialiasing',
        root_dir / 'inputs' / 'train' / 'mpl-generated-scatter',
    ]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    params = Params(
        name = args.name,
        logs_dir = root_dir / 'logs',
        device = device,
        overfit = args.overfit,
        clear_previous_results = args.clean_previous,
        clear_top_folder = False,
        num_frozen_epochs = args.epochs_decoder_only,
        num_finetune_epochs = args.epochs_finetune,
    )

    hparams = Hparams(
        batch_size = args.batch_size,
        encoder_lr = args.encoder_lr,
        decoder_lr = args.decoder_lr,
        loss = args.loss,
        regression_model = args.model,
    )

    data = DrawingsData(datasets_path, params, hparams)
    trainer = RegressionTrainer(params, hparams)
    trainer.train (data)

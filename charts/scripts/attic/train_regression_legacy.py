#!/usr/bin/env python3

from dlcharts.common.dataset import LabeledImage
from dlcharts.common.timer import Timer
import dlcharts.pytorch.color_regression as cr
import dlcharts.pytorch.utils as utils
from dlcharts.pytorch.utils import Experiment, num_trainable_parameters, is_google_colab, merge_dicts

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import transforms
import torch.profiler

import torch_lr_finder
import timm
from ptflops import get_model_complexity_info

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from icecream import ic
from tqdm import tqdm

import os
from pathlib import Path
import time
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
ic(use_cuda)

TEST_RUN = True

root_dir = Path(__file__).parent.parent
ic(root_dir)

preprocessor = cr.ImagePreprocessor(device, target_size=128)

dataset_path = Path("/content/datasets/drawings") if is_google_colab() else root_dir / 'inputs' / 'opencv-generated' / 'drawings'

dataset = cr.ColorRegressionImageDataset(dataset_path, preprocessor)
n_train = max(int(len(dataset) * 0.5), 1)
n_val = len(dataset) - n_train
# train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
generator = torch.Generator().manual_seed(42)

train_indices = range(0, n_train)
val_indices = range(n_train, len(dataset))

small_subset = TEST_RUN or not is_google_colab()
if small_subset:
    N = 16
    train_indices = random.sample(train_indices, N)
    val_indices = random.sample(val_indices, N)

train_sampler = SubsetRandomSampler(train_indices, generator=generator)
val_sampler = SubsetRandomSampler(val_indices, generator=generator)

DEFAULT_BATCH_SIZE=64 if is_google_colab() else 4
WORKERS=os.cpu_count() if is_google_colab() else 0

monitored_train_samples = random.sample(train_indices, 5)
monitored_val_samples = random.sample(val_indices, 5)
# monitored_sample = dataset[0]
# monitored_sample_inputs = torch.unsqueeze(monitored_sample[0], dim=0)
# monitored_samples_json = [

class Config:
    def __init__(self, name, batch_size=DEFAULT_BATCH_SIZE): 
        self.name = name
        self.batch_size = batch_size
    def create_net(self): return None
    def create_optimizer(self, net): return None
    def create_scheduler(self, optimizer, frozen, steps_per_epoch, total_epochs): return None
    def get_hyperparams(self): return dict(name=self.name, batch=self.batch_size)

net = None

def run_xp_config (xp: Experiment, config: Config, frozen_epochs: int, total_epochs: int, profiler = None):
    torch.cuda.empty_cache()
    global net # Make sure that we keep the last net to play with it after.    
    
    # Make sure that we release as much memory as possible
    net = None
    utils.clear_gpu_memory()

    net = config.create_net()
    net.to(device)
    optimizer = config.create_optimizer(net)

    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=config.batch_size, num_workers=WORKERS)
    val_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=config.batch_size, num_workers=WORKERS)
     
    criterion = nn.MSELoss()

    run_lr_finder = False
    if run_lr_finder:
        lr_finder = torch_lr_finder.LRFinder(net, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_dataloader, start_lr=1e-5, end_lr=1, num_iter=100)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state

    val_accuracy = 0.0
    training_loss = 0.0
    val_loss = 0.0

    def train (first_epoch, end_epoch, optimizer, scheduler):
        pbar = tqdm(range(first_epoch, end_epoch))
        for epoch in pbar:  # loop over the dataset multiple times
            nonlocal training_loss, val_loss, val_accuracy
            net.train()
            cumulated_training_loss = 0.0
            tstart = time.time()
            
            # batch_bar = tqdm(train_dataloader, leave=False)
            for i, data in enumerate(train_dataloader):
                inputs, labels, json_files = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                batch_loss = loss.item()
                # xp.writer.add_scalar("Single Batch Loss", batch_loss, epoch)

                cumulated_training_loss += batch_loss

                if scheduler:
                    scheduler.step()

                if profiler:
                    profiler.step()

            # Very important for batch norm layers.
            net.eval()

            def evaluate_images_at_indices(indices):                
                inputs = []
                outputs = []
                targets = []
                for idx in indices:
                    input, target = [x.to(device) for x in dataset[idx][:2]]
                    output = net(input.unsqueeze(0)).squeeze(0)
                    inputs.append(preprocessor.denormalize_and_clip_as_tensor(input.detach().cpu()))
                    outputs.append(preprocessor.denormalize_and_clip_as_tensor(output.detach().cpu()))
                    targets.append(preprocessor.denormalize_and_clip_as_tensor(target.detach().cpu()))
                return torch.cat([torch.cat(outputs, dim=2), torch.cat(targets, dim=2), torch.cat(inputs, dim=2)], dim=1)

            results_train = evaluate_images_at_indices(monitored_train_samples)
            xp.writer.add_image("Train Samples", results_train, epoch)

            results_val = evaluate_images_at_indices(monitored_val_samples)
            xp.writer.add_image("Val Samples", results_val, epoch)

            training_loss = cumulated_training_loss / len(train_dataloader)
            xp.writer.add_scalar("Training Loss", training_loss, epoch)
            
            val_loss = cr.compute_average_loss (val_dataloader, net, criterion, device)
            xp.writer.add_scalar("Validation Loss", val_loss, epoch)

            val_accuracy = cr.compute_accuracy (val_dataloader, net, criterion, device)
            xp.writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

            elapsedSecs = (time.time() - tstart)
            xp.writer.add_scalar("Elapsed Time (s)", elapsedSecs, epoch)
            # print(f"[{epoch}] [TRAIN_LOSS={training_loss:.4f}] [VAL_LOSS={val_loss:.4f}] [{elapsedSecs:.1f}s]")
            
            xp.writer.add_histogram("enc0", net.decoder.enc0.block[3].weight, global_step=epoch)
            xp.writer.add_histogram("dec0", net.decoder.dec0.block[3].weight, global_step=epoch)

            pbar.set_postfix({'train_loss': training_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

            if epoch > 0 and epoch % 10 == 0:
                xp.save_checkpoint(epoch)

        xp.save_checkpoint(end_epoch-1)

    net.freeze_encoder()
    ic(num_trainable_parameters(net))
    scheduler = config.create_scheduler(optimizer, frozen=True, steps_per_epoch=len(train_dataloader), total_epochs=frozen_epochs)
    xp.prepare (config.name + '-frozen', net, optimizer, scheduler, device, dataset[0][0].unsqueeze(0).to(device))
    xp.writer.add_text("Model Complexity", ", ".join(get_model_complexity_info(net, (3, 128, 128), as_strings=True, print_per_layer_stat=False, verbose=False)), global_step=None, walltime=None)
    train(xp.first_epoch, frozen_epochs, optimizer, scheduler)
    xp.finalize(hparams = config.get_hyperparams(), metrics={'hparam/train_loss': training_loss, 'hparam/val_loss': val_loss, 'hparam/accuracy': val_accuracy})

    utils.clear_gpu_memory()

    net.unfreeze_encoder()
    scheduler = config.create_scheduler(optimizer, frozen=False, steps_per_epoch=len(train_dataloader), total_epochs=(total_epochs - frozen_epochs))
    xp.prepare (config.name + '-tune', net, optimizer, scheduler, device, dataset[0][0].unsqueeze(0).to(device), default_first_epoch=frozen_epochs)
    ic(num_trainable_parameters(net))
    train(xp.first_epoch, total_epochs, optimizer, scheduler)
    xp.finalize(hparams = config.get_hyperparams(), metrics={'hparam/train_loss': training_loss, 'hparam/val_loss': val_loss, 'hparam/accuracy': val_accuracy})

    print('Finished Training!')
    utils.clear_gpu_memory()

class ConfigUnet1Adam(Config):
    def __init__(self, name, max_lr_frozen, max_lr_tune, one_cycle: bool = True, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.one_cycle = one_cycle
        self.max_lr_frozen = max_lr_frozen
        self.max_lr_tune = max_lr_tune

    def create_net(self): return cr.RegressionNet_Unet1()

    def create_optimizer(self, net): 
        return optim.Adam([
            {'params': net.encoder.parameters(), 'lr': self.max_lr_frozen[0] },
            {'params': net.decoder.parameters(), 'lr': self.max_lr_frozen[1] }
        ])

    def create_scheduler(self, optimizer, frozen: bool, steps_per_epoch: int, total_epochs: int):
        if not self.one_cycle:
            return None
        max_lr = self.max_lr_frozen if frozen else self.max_lr_tune
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, steps_per_epoch=steps_per_epoch, epochs=total_epochs)

    def get_hyperparams(self):
        return merge_dicts(super().get_hyperparams(), dict(
            net='unet1',
            opt='adam',
            sched='1cycle' if self.one_cycle else 'none',
            enc_lr_frozen=self.max_lr_frozen[0],
            dec_lr_frozen=self.max_lr_frozen[1],
            enc_lr_tune=self.max_lr_tune[0],
            dec_lr_tune=self.max_lr_tune[1]
        ))

class ConfigUnet1AdamW(ConfigUnet1Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_optimizer(self, net): 
        return optim.AdamW([
            {'params': net.encoder.parameters(), 'lr': self.max_lr_frozen[0] },
            {'params': net.decoder.parameters(), 'lr': self.max_lr_frozen[1] }
        ])

    def get_hyperparams(self):
        return merge_dicts(super().get_hyperparams(), dict(opt='AdamW'))

class ConfigUnetResAdamW(ConfigUnet1AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_net(self):
        return cr.RegressionNet_Unet1(residual_mode = True)

    def get_hyperparams(self):
        return merge_dicts(super().get_hyperparams(), dict(net='unetres'))

class ConfigUnet1SGD(ConfigUnet1Adam):
    def __init__(self, name, momentum, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.momentum = momentum
    
    def create_optimizer(self, net):
        return optim.SGD([
            {'params': net.encoder.parameters(), 'lr': self.max_lr_frozen[0] },
            {'params': net.decoder.parameters(), 'lr': self.max_lr_frozen[1] }
        ], momentum=self.momentum)

    def get_hyperparams(self):
        return merge_dicts(super().get_hyperparams(), dict(opt='SGD'))

configs = [
    # TODO: test diffent batch sizes and lr with AdamW
    # ConfigUnet1Adam('unet1_adam_3e4', max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 3e-4), one_cycle=False),

    # ConfigUnet1Adam('unet1_adam_1cycle_1e3', max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1Adam('unet1_adam_1cycle_3e4', max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 1e-4)),
    
    # ConfigUnet1SGD('unet1_sgd_1cycle_1e3_09', max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4), momentum=0.9),
    # ConfigUnet1SGD('unet1_sgd_1cycle_1e3_099', max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4), momentum=0.99),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_1e3_3e4', max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_3e4_1e4', max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 1e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_3e4_3e4', max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_1e3_1e3', max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 1e-3)),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn128_1e3_3e4', batch_size=128, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_1e3_3e4', batch_size=64, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn32_1e3_3e4', batch_size=32, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn16_1e3_3e4', batch_size=16, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_3e4_3e4', batch_size=64, max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_1e3', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(1e-5, 1e-3)),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_3e4_3e4_1e5', batch_size=64, max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_1e3_3e4_1e5', batch_size=64, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-5, 3e-4)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_1e3_1e5', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(1e-5, 1e-3)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_1e2_5e3_1e5', batch_size=64, max_lr_frozen=(1e-5, 1e-2), max_lr_tune=(1e-5, 5e-3)),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_3e4_3e4_1e4', batch_size=64, max_lr_frozen=(1e-5, 3e-4), max_lr_tune=(1e-4, 3e-4)),

    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_1e3_1e4', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(1e-4, 1e-3)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_1e3_1e3', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(1e-3, 1e-3)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_1e3_5e3', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(5e-3, 1e-3)),
    # ConfigUnet1AdamW('unet1_adamw_1cycle_bn64_5e3_5e3_5e3', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(5e-3, 5e-3)),

    ConfigUnetResAdamW('unetres_adamw_1cycle_bn64_5e3_1e3_1e4', batch_size=64, max_lr_frozen=(1e-5, 5e-3), max_lr_tune=(1e-4, 1e-3)),
    ConfigUnetResAdamW('unetres_adamw_1cycle_bn64_1e3_1e3_1e4', batch_size=64, max_lr_frozen=(1e-5, 1e-3), max_lr_tune=(1e-4, 1e-3)),
]

# with torch.profiler.profile(
#     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(
#         wait=2,
#         warmup=2,
#         active=6,
#         repeat=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler(utils.default_output_dir / 'profiler'),
#     with_stack=True
# ) as profiler:
#     xp = Experiment("2022-Feb01-CR1-Profiler", utils.default_output_dir, clear_previous_results=True, clear_top_folder=True)
#     print (f"=== RUNNING PROFILING CONFIG {configs[0].name} ==")
#     run_xp_config (xp, configs[0], frozen_epochs=1, total_epochs=1, profiler=profiler)

logs_root_dir = utils.default_output_dir
xp = Experiment("2022-Feb04-UnetRes" + ('TESTRUN' if TEST_RUN else ''), 
                logs_root_dir, 
                clear_previous_results=False, 
                clear_top_folder=False)
for i, config in enumerate(configs):
    print (f"=== [{i+1}/{len(configs)}] RUNNING CONFIG {config.name} ==")
    if TEST_RUN:
        run_xp_config (xp, config, frozen_epochs=2, total_epochs=4)
    else:
        run_xp_config (xp, config, frozen_epochs=40, total_epochs=60)

def load_specific_checkpoint (name):
    checkpoint = torch.load(xp.log_path / name, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# load_specific_checkpoint ("checkpoint-00701.pt")
# torch.save (net, "regression_unet_v1.pt")

with torch.no_grad():
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=1, num_workers=0)
    input, labels, _ = next(iter(train_dataloader))
    input, labels = [x.to(device) for x in [input, labels]]
    output = net(input.to(device))
    #clear_output(wait=True)
    plt.figure()
    plt.imshow (preprocessor.denormalize_and_clip_as_numpy(output[0]))
    plt.figure()
    plt.imshow (preprocessor.denormalize_and_clip_as_numpy(labels[0]))
    plt.figure()
    plt.imshow (preprocessor.denormalize_and_clip_as_numpy(input[0]))

# Stop the google colab VM
if not TEST_RUN:
    utils.stop_google_colab_vm ()

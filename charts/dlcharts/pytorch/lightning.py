from ..common.utils import InfiniteIterator

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import pytorch_lightning.callbacks.progress.tqdm_progress as tqdm_progress

import sys
from typing import Optional

class GlobalProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__(process_position=1)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.global_progress_bar = tqdm_progress.Tqdm(
            desc=f"Overall Training ({trainer.max_epochs} epochs)",
            initial=trainer.current_epoch,
            position=1,
            disable=False,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
               
        self.total_batches_per_epoch = self.total_train_batches

        tqdm_progress.reset(self.global_progress_bar, total=trainer.max_epochs*self.total_batches_per_epoch, current=trainer.current_epoch)
        super().on_train_start(trainer, pl_module)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = trainer.current_epoch*self.total_batches_per_epoch + self.train_batch_idx
        if self._should_update(n):
            self._update_bar(self.global_progress_bar)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

class ValidationStepCallback(pl.Callback):
    def __init__(self, datamodule: pl.LightningDataModule):
        self.datamodule = datamodule
        self.infinite_iterator = None

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        # This is somehow very important to avoid multiple workers to hang.
        del self.infinite_iterator
        return super().teardown(trainer, pl_module, stage)

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

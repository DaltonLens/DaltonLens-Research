import torch
import torch.nn
import torch.optim

from torch.utils.data import Dataset, ConcatDataset, Sampler, BatchSampler, SubsetRandomSampler

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from pathlib import Path
import shutil
from contextlib import contextmanager

import gc
import inspect
import sys
from typing import Dict, Sequence, List, Iterator

default_output_dir = Path(__file__).resolve().parent / "experiments"

_already_checked_is_google_colab = None

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    gettrace = getattr(sys, 'gettrace', lambda : None) 
    return gettrace() is not None

def is_google_colab():
    global _already_checked_is_google_colab
    if _already_checked_is_google_colab is not None:
        return _already_checked_is_google_colab
    try:
        import google.colab
        _already_checked_is_google_colab = True
    except:
        _already_checked_is_google_colab = False
    return _already_checked_is_google_colab

def stop_google_colab_vm():
    if is_google_colab():
        import subprocess
        subprocess.run(["touch", "/content/stop_colab_vm"])
        subprocess.run(["jupyter", "notebook", "stop", "8888"])
        subprocess.run(["sleep", "5"])
        subprocess.run(["kill", "-9", "-1"])

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

class Experiment:
    def __init__(self, 
                 name: str,
                 logs_root_dir: Path = default_output_dir,
                 clear_previous_results: bool = False,
                 clear_top_folder=False):
        assert len(name) >= 1
        self.root_log_path = logs_root_dir / name        
        print (f"[XP] storing experiment data to {self.root_log_path}")
        self.clear_previous_results = clear_previous_results
        if clear_top_folder and self.root_log_path.exists():
            print (f"Warning: removing the top-level {self.root_log_path}")
            shutil.rmtree (self.root_log_path)

    def prepare(self, 
                config_name: str,
                net: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler],
                device: torch.device,
                sample_input: torch.Tensor,
                default_first_epoch: int = 0):
        assert len(config_name) > 3
        self.log_path = self.root_log_path / config_name
        self.net = net
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.first_epoch = default_first_epoch

        print (f"[XP] storing config data to {self.log_path}")

        if self.clear_previous_results and self.log_path.exists():
            print (f"Warning: removing the existing {self.log_path}")
            shutil.rmtree (self.log_path)
        self.log_path.mkdir (parents=True, exist_ok=True)

        checkpoints = list(sorted(self.log_path.glob("checkpoint-*.pt")))
        checkpoint = checkpoints[-1] if checkpoints else None
        if checkpoint:
            print (f"Loading checkpoint {checkpoint}")
            checkpoint = torch.load(checkpoint, map_location=device)
            self.first_epoch = checkpoint['last_epoch'] + 1
            net.load_state_dict(checkpoint['model_state_dict'])
            if schedulers:
                for name, scheduler in schedulers.items():
                    # Hack to skip the fine tune scheduler in case we changed the number
                    # of finetune iterations.
                    if True: # name != 'finetune_scheduler':
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'][name])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            net.train()

        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.add_graph(net, sample_input)

    def log_lr (self, optimizer: torch.optim.Optimizer, global_step: int):
        for idx, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'scheduler/lr-{idx}', param_group['lr'], global_step)
            if 'momentum' in param_group:
                self.writer.add_scalar(f'scheduler/momentum-{idx}', param_group['momentum'], global_step)
            elif 'betas' in param_group: # for Adam
                self.writer.add_scalar(f'scheduler/momentum-{idx}', param_group['betas'][0], global_step)

    def finalize(self, hparams, metrics):
        self.writer.add_hparams(hparams, metrics)

    def save_checkpoint (self, epoch):
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        scheduler_state_dicts = {}
        if self.schedulers:
            for name, scheduler in self.schedulers.items():
                scheduler_state_dicts[name] = scheduler.state_dict()
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dicts,
            'last_epoch': epoch,
        }, self.log_path / f"checkpoint-{epoch:05d}.pt")


def num_trainable_parameters(model):
    return pretty_size(sum(p.numel() for p in model.parameters() if p.requires_grad))

def get_model_complexity_info(model, input_size=(3, 192, 192), device='cpu'):
    import ptflops
    print (ptflops.get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False, verbose=False))
    import torchsummary
    print (torchsummary.summary(model, input_size, device=device))

def clear_gpu_memory():    
    gc.collect()
    torch.cuda.empty_cache()

def find_names(obj):
    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names

def show_gpu_memory_with_names():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), find_names(obj))
        except:
            pass

def pretty_size(num, suffix=""):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"

def show_gpu_memory():
    """Prints a list of the Tensors being tracked by the garbage collector."""
    gpu_only = True
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s: %s" % (type(obj).__name__, 
                                          str(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.data.is_cuda:
                    print("%s → %s: %s" % (type(obj).__name__, 
                                                   type(obj.data).__name__, 
                                                   str(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass
    print("Total size:", pretty_size(total_size))

class ClusteredDataset(Dataset):
    datasets_per_cluster: List
    range_per_cluster: List

    def __init__(self, datasets: List[Dataset]):
        super().__init__()
        concat_dataset_per_cluster = {}       
        for dataset in datasets:
            assert hasattr(dataset, 'cluster_index')
            if dataset.cluster_index not in concat_dataset_per_cluster:
                concat_dataset_per_cluster[dataset.cluster_index] = []
            concat_dataset_per_cluster[dataset.cluster_index].append(dataset)
               
        self.datasets_per_cluster = []
        self.range_per_cluster = []
        last_index = 0
        for datasets in concat_dataset_per_cluster.values():
            concat_dataset = ConcatDataset(datasets)
            n = len(concat_dataset)
            self.datasets_per_cluster.append(concat_dataset)
            self.range_per_cluster.append((last_index, last_index + n))
            last_index += n

    def __len__(self):
        return self.range_per_cluster[-1][1]

    def __getitem__(self, idx):
        # Option: use bisect like ConcatDataset if the number of dataset ever becomes very large.
        for cluster_index, r in enumerate(self.range_per_cluster):
            if idx >= r[0] and idx < r[1]:
                return self.datasets_per_cluster[cluster_index][idx - r[0]]
        return None

class SubsetSampler(Sampler):
    def __init__ (self, indices: Sequence[int]):
        self.indices = indices
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
    
class ClusteredBatchRandomSampler(Sampler):
    def __init__ (self, clustered_dataset: ClusteredDataset, batch_size: int, shuffle: bool, drop_last: bool = False, generator=None):
        self.generator = generator
        self.clustered_batch_samplers = []
        for cluster_range, dataset in zip(clustered_dataset.range_per_cluster, clustered_dataset.datasets_per_cluster):
            if shuffle:
                sampler = SubsetRandomSampler(range(cluster_range[0], cluster_range[1]))
            else:
                sampler = SubsetSampler(range(cluster_range[0], cluster_range[1]))
            self.clustered_batch_samplers.append (BatchSampler(sampler, batch_size, drop_last))

        self.batch_sampler_indices = []
        for idx, batch_sampler in enumerate(self.clustered_batch_samplers):
            self.batch_sampler_indices += [idx] * len(batch_sampler)        
    
    def __iter__(self):
        per_cluster_iterators = [b.__iter__() for b in self.clustered_batch_samplers]
        for idx in torch.randperm(len(self.batch_sampler_indices), generator=self.generator):
            sample_idx = self.batch_sampler_indices[idx]
            yield next(per_cluster_iterators[sample_idx])
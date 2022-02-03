import torch
import torch.nn
import torch.optim

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from pathlib import Path
import shutil

import gc
import inspect

default_output_dir = Path(__file__).resolve().parent / "experiments"

_already_checked_is_google_colab = None

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
        subprocess.run(["kill", "-9", "-l"])

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
        assert len(name) > 3
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
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                sample_input: torch.Tensor,
                default_first_epoch: int = 0):
        assert len(config_name) > 3
        self.log_path = self.root_log_path / config_name
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
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
            if scheduler:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            net.train()

        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.add_graph(net, sample_input)

    def finalize(self, hparams, metrics):
        self.writer.add_hparams(hparams, metrics)

    def save_checkpoint (self, epoch):
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else {},
            'last_epoch': epoch,
        }, self.log_path / f"checkpoint-{epoch:05d}.pt")


def num_trainable_parameters(model):
    return pretty_size(sum(p.numel() for p in model.parameters() if p.requires_grad))

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
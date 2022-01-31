import torch
import torch.nn
import torch.optim

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from pathlib import Path
import shutil

default_output_dir = Path(__file__).resolve().parent / "experiments"

class Experiment:
    def __init__(self, name: str,
                 logs_root_dir: Path = default_output_dir,
                 clear_previous_results: bool = False):
        assert len(name) > 3
        self.name = str
        self.log_path = logs_root_dir / name
        print (f"Will store the experiment data to {self.log_path}")
        self.clear_previous_results = clear_previous_results
        self.first_epoch = 0

    def prepare (self, net: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, sample_input: torch.Tensor):
        self.net = net
        self.optimizer = optimizer

        if self.clear_previous_results and self.log_path.exists():
            print (f"Warning: removing the existing {self.log_path}")
            shutil.rmtree (self.log_path)            
        self.log_path.mkdir (parents=True, exist_ok=True)

        checkpoints = list(sorted(self.log_path.glob("checkpoint-*.pt")))
        checkpoint = checkpoints[-1] if checkpoints else None
        if checkpoint:
            print (f"Loading checkpoint {checkpoint}")
            checkpoint = torch.load(checkpoint, map_location=device)
            self.first_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            net.train()

        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.add_graph(net, sample_input)

    def save_checkpoint (self, epoch):
        # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, self.log_path / f"checkpoint-{epoch:05d}.pt")


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

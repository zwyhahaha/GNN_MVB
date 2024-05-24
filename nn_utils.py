import torch
from torch import Tensor
from torch.nn import Sequential, Linear, Dropout, Parameter, LogSoftmax, Sigmoid, ReLU, LeakyReLU, Tanh, SELU, SiLU
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing, MultiAggregation, global_mean_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm, InstanceNorm
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax

from typing import List, Union, Optional
import numpy as np
import pandas as pd

class NoNorm(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(NoNorm, self).__init__()

    def __call__(self, x, *args, **kwargs):
        return x

activations = {'sigmoid': Sigmoid, 'relu': ReLU, 'leakyrelu':LeakyReLU, 'tanh': Tanh, 'selu': SELU,  'silu': SiLU}
normalizations = {'batch': BatchNorm, 'graph': GraphNorm, 'instance': InstanceNorm, 'nonorm':NoNorm}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unit_act(x):
    return x

def reset_parameters_(model):
    if isinstance(model.act(), SELU):
        keys = list(model.state_dict().keys())
        for i, param in enumerate(model.parameters()):
            # biases zero
            param_type = keys[i].split('.')[-1]

            if param_type in ['bias', 'eps']:
                torch.nn.init.constant_(param, 0.0)
            # others using lecun-normal initialization
            else:
                try:
                    torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
                except:
                    torch.nn.init.normal_(param)
    else:
        for module in model.children():
            if list(module.children()):
                for submodule in module.children():
                    if list(submodule.children()):
                        for subsubmodule in submodule.children():
                            if subsubmodule._parameters:
                                subsubmodule.reset_parameters()
                    elif len(submodule._parameters) > 0:
                        submodule.reset_parameters()
            else:
                if module._parameters:
                    module.reset_parameters()

def get_var_and_con_batch_idx(batch, num_var_nodes, num_con_nodes, device=DEVICE):
    
    if batch.batch is None: # i.e., batch is a single pytorch data object
        var_batch_idx = torch.zeros(num_var_nodes, dtype=torch.long, device=device)
        con_batch_idx = torch.zeros(num_con_nodes, dtype=torch.long, device=device)
    else:
        var_batch_idx = batch.batch

    present_con_indices = torch.unique(batch.index_con)
    new_con_indices = torch.arange(present_con_indices.size(0))
    indmap = pd.Series(dict(zip(present_con_indices.cpu().numpy(), new_con_indices.cpu().numpy())))
    con_batch_idx = torch.tensor(indmap[batch.index_con.cpu().numpy()].values, device=DEVICE)

    return var_batch_idx, con_batch_idx

class PreNormException(Exception):
    pass

class PreNormLayer(torch.nn.Module):
    """
    source: https://github.com/ds4dm/learn2branch-ecole/blob/main/model/model.py
    """
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False

class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        # assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
        #        len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)
        assert self.num_lrs == 1
        self.optimizer = optimizer
        self.warmup_epochs = np.array([warmup_epochs])
        self.total_epochs = np.array([total_epochs])
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array([init_lr])
        self.max_lr = np.array([max_lr])
        self.final_lr = np.array([final_lr])

        self.current_step = 0
        self.lr = [init_lr]
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]

import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple, List, Union, Dict, cast
from torch.utils.data import DataLoader
from CIFAR.models.utils import StraightThrough, AvgPoolConv
from distributed_utils.dist_helper import allaverage


# ------------------------- New Version ---------------------------


class SpikeModule(nn.Module):
    """
    Spike-based Module that can handle spatial-temporal information.
    """
    def __init__(self, sim_length: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = True):
        super(SpikeModule, self).__init__()
        if isinstance(conv, nn.Conv2d):
            self.fwd_kwargs = {"stride": conv.stride, "padding": conv.padding,
                               "dilation": conv.dilation, "groups": conv.groups}
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = {}
            self.fwd_func = F.linear
        self.threshold = None
        self.mem_pot = 0
        self.mem_pot_init = 0
        self.weight = conv.weight
        self.org_weight = copy.deepcopy(conv.weight.data)
        if conv.bias is not None:
            self.bias = conv.bias
            self.org_bias = copy.deepcopy(conv.bias.data)
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the spike forward default
        self.use_spike = False
        self.enable_shift = enable_shift
        self.sim_length = sim_length
        self.cur_t = 0
        self.relu = StraightThrough()

    def forward(self, input: torch.Tensor):
        if self.use_spike:
            self.cur_t += 1
            x = self.fwd_func(input, self.weight, self.bias, **self.fwd_kwargs)
            if self.enable_shift is True and self.threshold is not None:
                x = x + self.threshold * 0.5 / self.sim_length
            self.mem_pot = self.mem_pot + x
            spike = (self.mem_pot >= self.threshold).float() * self.threshold
            self.mem_pot -= spike
            return spike
        else:
            return self.relu(self.fwd_func(input, self.org_weight, self.org_bias, **self.fwd_kwargs))

    def init_membrane_potential(self):
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.cur_t = 0


class SpikeModel(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.spike_module_refactor(self.model, sim_length)
        self.use_spike = False

        assert sim_length > 0, "SNN does not accept negative simulation length"
        self.T = sim_length

    def spike_module_refactor(self, module: nn.Module, sim_length: int, prev_module=None):
        """
        Recursively replace the normal conv2d to SpikeConv2d

        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param sim_length: simulation length, aka total time steps
        :param prev_module: use this to add relu to prev_spikemodule
        """
        prev_module = prev_module
        for name, immediate_child_module in module.named_children():
            if type(immediate_child_module) in self.specials:
                setattr(module, name, self.specials[type(immediate_child_module)]
                                                        (immediate_child_module, sim_length=sim_length))
            elif isinstance(immediate_child_module, nn.Conv2d) and not isinstance(immediate_child_module, AvgPoolConv):
                setattr(module, name, SpikeModule(sim_length=sim_length, conv=immediate_child_module))
                prev_module = getattr(module, name)
            elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
                if prev_module is not None:
                    prev_module.add_module('relu', immediate_child_module)
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(immediate_child_module, AvgPoolConv):
                relu = immediate_child_module.relu
                setattr(module, name, SpikeModule(sim_length=sim_length, conv=immediate_child_module))
                getattr(module, name).add_module('relu', relu)
            else:
                prev_module = self.spike_module_refactor(immediate_child_module, sim_length=sim_length, prev_module=prev_module)

        return prev_module

    def set_spike_state(self, use_spike: bool = True):
        self.use_spike = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.use_spike = use_spike

    def init_membrane_potential(self):
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.init_membrane_potential()

    def forward(self, input):
        if self.use_spike:
            self.init_membrane_potential()
            out = 0
            for sim in range(self.T):
                out += self.model(input)
        else:
            out = self.model(input)
        return out


# ------------------------- Max Activation ---------------------------


class DataSaverHook:
    def __init__(self, momentum: Union[float, None] = 0.9, sim_length: int = 8,
                 mse: bool = True, percentile: Union[float, None] = None, channel_wise: bool = False,
                 dist_avg: bool = False):
        self.momentum = momentum
        self.max_act = None
        self.T = sim_length
        self.mse = mse
        self.percentile = percentile
        self.channel_wise = channel_wise
        self.dist_avg = dist_avg

    def __call__(self, module, input_batch, output_batch):
        def get_act_thresh(tensor):
            if self.mse:
                act_thresh = find_threshold_mse(output_batch, T=self.T, channel_wise=self.channel_wise)
            elif self.percentile is not None:
                assert 0. <= self.percentile <= 1.0
                act_thresh = quantile(output_batch, self.percentile)
            else:
                act_thresh = tensor.max()
            return act_thresh

        if self.max_act is None:
            self.max_act = get_act_thresh(output_batch)
        else:
            cur_max = get_act_thresh(output_batch)
            if self.momentum is None:
                self.max_act = self.max_act if self.max_act > cur_max else cur_max
            else:
                self.max_act = self.momentum * self.max_act + (1 - self.momentum) * cur_max
        if self.dist_avg:
            allaverage(self.max_act)
        module.threshold = self.max_act


def quantile(tensor: torch.Tensor, p: float):
    try:
        return torch.quantile(tensor, p)
    except:
        tensor_np = tensor.cpu().detach().numpy()
        return torch.tensor(np.percentile(tensor_np, q=p*100)).type_as(tensor)


def find_threshold_mse(tensor: torch.Tensor, T: int = 8, channel_wise: bool = True):
    """
    This function use grid search to find the best suitable
    threshold value for snn.
    :param tensor: the output batch tensor,
    :param T: simulation length
    :param channel_wise: set threshold channel-wise
    :return: threshold with MMSE
    """
    def clip_floor(tensor:torch.Tensor, T: int, Vth: Union[float, torch.Tensor]):
        snn_out = torch.clamp(tensor / Vth * T, min=0, max=T)
        return snn_out.floor() * Vth / T

    if channel_wise:
        num_channel =tensor.shape[1]
        best_Vth = torch.ones(num_channel).type_as(tensor)
        # determine the Vth channel-by-channel
        for i in range(num_channel):
            best_Vth[i] = find_threshold_mse(tensor[:, i], T, channel_wise=False)
        best_Vth = best_Vth.reshape(1, num_channel, 1, 1) if len(tensor.shape)==4 else best_Vth.reshape(1, num_channel)
    else:
        max_act = tensor.max()
        best_score = 1e5
        best_Vth = 0
        for i in range(95):
            new_Vth = max_act * (1.0 - (i * 0.01))
            mse = lp_loss(tensor, clip_floor(tensor, T, new_Vth), p=2.0, reduction='other')
            if mse < best_score:
                best_Vth = new_Vth
                best_score = mse

    return best_Vth


@torch.no_grad()
def get_maximum_activation(train_loader: torch.utils.data.DataLoader,
                           model: SpikeModel,
                           momentum: Union[float, None] = 0.9,
                           iters: int = 20,
                           sim_length: int = 8,
                           mse: bool = True, percentile: Union[float, None] = None,
                           channel_wise: bool = False,
                           dist_avg: bool = False):
    """
    This function store the maximum activation in each convolutional or FC layer.
    :param train_loader: Data loader of the training set
    :param model: target model
    :param momentum: if use momentum, the max activation will be EMA updated
    :param iters: number of iterations to calculate the max act
    :param sim_length: sim_length when computing the mse of SNN output
    :param mse: if Ture, use MMSE to find the V_th
    :param percentile: if mse = False and percentile is in [0,1], use percentile to find the V_th
    :param channel_wise: use channel-wise mse
    :param dist_avg: if True, then compute mean between distributed nodes
    :return: model with stored max activation buffer
    """
    # do not use train mode here (avoid bn update)
    model.set_spike_state(use_spike=False)
    model.eval()
    device = next(model.parameters()).device
    hook_list = []
    for m in model.modules():
        if isinstance(m, SpikeModule):
            hook_list += [m.register_forward_hook(DataSaverHook(momentum, sim_length, mse, percentile, channel_wise,
                                                                dist_avg))]
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device=device)
        _ = model(input)
        if i > iters:
            break
    for h in hook_list:
        h.remove()


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    elif reduction == 'channel_split':
        return (pred-tgt).abs().pow(p).sum((0,2,3))
    else:
        return (pred-tgt).abs().pow(p).mean()

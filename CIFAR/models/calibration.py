import torch
from torch.utils.data import DataLoader
from CIFAR.models.spiking_layer import SpikeModule, SpikeModel, lp_loss
from distributed_utils.dist_helper import allaverage, allreduce
from CIFAR.models.resnet import SpikeResModule as SpikeResModule_CIFAR


def bias_corr_model(model: SpikeModel, train_loader: torch.utils.data.DataLoader, correct_mempot: bool = False,
                    dist_avg: bool = False):
    """
    This function corrects the bias or potential in SNN, by matching the
    activation expectation in some training set samples.
    Here we only sample one batch of the training set.

    :param model: SpikeModel that need to be corrected with bias
    :param train_loader: Training images
    :param correct_mempot: if True, the correct the initial membrane potential
    :param dist_avg: if True, then average the tensor between distributed GPUs
    :return: SpikeModel with corrected bias
    """
    device = next(model.parameters()).device
    for (input, target) in train_loader:
        input = input.to(device=device)
        # begin bias correction layer-by-layer
        for name, module in model.named_modules():
            if isinstance(module, SpikeModule):
                print('\nEmpirical Bias Correction for layer {}:'.format(name) if not correct_mempot else
                      '\nEmpirical Potential Correction for layer {}:'.format(name))
                emp_bias_corr(model, module, input, correct_mempot, dist_avg)
        # only perform BC for one time
        break


def emp_bias_corr(model: SpikeModel, module: SpikeModule, train_data: torch.Tensor, correct_mempot: bool = False,
                  dist_avg: bool = False):
    # compute the original output
    model.set_spike_state(use_spike=False)
    get_out = GetLayerInputOutput(model, module)
    org_out = get_out(train_data)[1]
    # # clip output here
    # org_out = org_out
    # compute the SNN output
    model.set_spike_state(use_spike=True)
    get_out.data_saver.reset()
    snn_out = get_out(train_data)[1]
    # divide the snn output by T
    snn_out = snn_out / model.T
    if not correct_mempot:
        # calculate the bias
        org_mean = org_out.mean(3).mean(2).mean(0).detach() if len(org_out.shape) == 4 else org_out.mean(0).detach()
        snn_mean = snn_out.mean(3).mean(2).mean(0).detach() if len(snn_out.shape) == 4 else snn_out.mean(0).detach()
        bias = (snn_mean - org_mean).data.detach()

        if dist_avg:
            allaverage(bias)
        # now let's absorb it.
        if module.bias is None:
            module.bias = - bias
        else:
            module.bias.data = module.bias.data - bias
        print((-bias).mean(), (-bias).var())
    else:
        # calculate the mean along the batch dimension
        org_mean, snn_mean = org_out.mean(0, keepdim=True), snn_out.mean(0, keepdim=True)
        pot_init_temp = ((org_mean - snn_mean) * model.T).data.detach()
        if dist_avg:
            allaverage(pot_init_temp)
        module.mem_pot_init = pot_init_temp


class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """
    def __init__(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch
        else:
            self.stored_output = output_batch + self.stored_output
        if self.stored_input is None:
            self.stored_input = input_batch[0]
        else:
            self.stored_input = input_batch[0] + self.stored_input
        if len(input_batch) == 2:
            if self.stored_residual is None:
                self.stored_residual = input_batch[1].detach()
            else:
                self.stored_residual = input_batch[1].detach() + self.stored_residual
        else:
            if self.stored_residual is None:
                self.stored_residual = 0

    def reset(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None


class GetLayerInputOutput:
    def __init__(self, model: SpikeModel, target_module: SpikeModule):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        # do not use train mode here (avoid bn update)
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        # note that we do not have to check model spike state here,
        # because the SpikeModel forward function can already do this
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), self.data_saver.stored_output.detach(), \
               self.data_saver.stored_residual


def floor_ste(x):
    return (x.floor() - x).detach() + x


def weights_cali_model(model: SpikeModel, train_loader: torch.utils.data.DataLoader,
                       learning_rate: float = 4e-5, optimize_iter: int = 5000,
                       batch_size: int = 32, num_cali_samples: int = 1024, dist_avg: bool = False):
    """
    This function calibrate the weights in SNN.

    :param model: SpikeModel that need to be corrected with bias
    :param train_loader: Training images data loader
    :param learning_rate: the learning rate of WC
    :param optimize_iter: the total iteration number of WC for each layer
    :param batch_size: mini-batch size for WC
    :param num_cali_samples: total sample number of WC
    :param dist_avg: if True, then average the tensor between distributed GPUs
    :return: SpikeModel with corrected bias
    """
    data_sample = []
    for (input, target) in train_loader:
        data_sample += [input]
        if len(data_sample) * data_sample[-1].shape[0] >= num_cali_samples:
            break
    data_sample = torch.cat(data_sample, dim=0)[:num_cali_samples]
    # begin weights calibration layer-by-layer

    for name, module in model.named_modules():
        if isinstance(module, (SpikeResModule_CIFAR)):
            print('\nEmpirical Weights Calibration for layer {}:'.format(name))
            weights_cali_res_layer(model, module, data_sample, learning_rate, optimize_iter,
                                   batch_size, num_cali_samples, dist_avg=dist_avg)
        elif isinstance(module, SpikeModule):
            print('\nEmpirical Weights Calibration for layer {}:'.format(name))
            weights_cali_layer(model, module, data_sample, learning_rate, optimize_iter,
                               batch_size, num_cali_samples, dist_avg=dist_avg)


def weights_cali_layer(model: SpikeModel, module: SpikeModule, data_sample: torch.Tensor,
                       learning_rate: float = 1e-5, optimize_iter: int = 10000,
                       batch_size: int = 32, num_cali_samples: int = 1024, keep_gpu: bool = True,
                       loss_func=lp_loss, dist_avg: bool = False):

    get_out = GetLayerInputOutput(model, module)
    device = next(module.parameters()).device
    data_sample = data_sample.to(device)
    cached_batches = []
    for i in range(int(data_sample.size(0) / batch_size)):
        # compute the original output
        model.set_spike_state(use_spike=False)
        _, cur_out, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        # compute the spike input
        model.set_spike_state(use_spike=True)
        cur_inp, _, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    # divide by T here
    cached_inps = cached_inps / model.T
    cached_outs = torch.cat([x[1] for x in cached_batches])

    del cached_batches
    torch.cuda.empty_cache()

    if keep_gpu:
        # Put all cached data on GPU for faster optimization
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)

    # build optimizer and lr scheduler
    optimizer = torch.optim.SGD([module.weight], lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iter, eta_min=0.)

    model.set_spike_state(use_spike=True)
    # start optimization
    for i in range(optimize_iter):
        idx = torch.randperm(num_cali_samples)[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        optimizer.zero_grad()
        module.zero_grad()

        snn_out = module.fwd_func(cur_inp, module.weight, module.bias, **module.fwd_kwargs)
        snn_out = torch.clamp(snn_out / module.threshold * model.T, min=0, max=model.T)
        snn_out = floor_ste(snn_out) * module.threshold / model.T
        err = loss_func(snn_out, cur_out)
        err.backward()
        if dist_avg:
            allreduce(module.weight.grad)
        optimizer.step()
        scheduler.step()

        if i % 300 == 0:
            print('Iteration {}\tLoss: {},'.format(i, err.item()))

    print("\n")


def weights_cali_res_layer(model: SpikeModel, module: SpikeModule, data_sample: torch.Tensor,
                           learning_rate: float = 1e-5, optimize_iter: int = 10000,
                           batch_size: int = 32, num_cali_samples: int = 1024, keep_gpu: bool = True,
                           loss_func=lp_loss, dist_avg: bool = False):

    get_out = GetLayerInputOutput(model, module)
    device = next(module.parameters()).device
    data_sample = data_sample.to(device)
    cached_batches = []
    for i in range(int(data_sample.size(0) / batch_size)):
        # compute the original output
        model.set_spike_state(use_spike=False)
        _, cur_out, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        # compute the spike input
        model.set_spike_state(use_spike=True)
        cur_inp, _, cur_res = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_res.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_inps = cached_inps / model.T
    cached_ress = torch.cat([x[2] for x in cached_batches])
    cached_ress = cached_ress / model.T
    cached_outs = torch.cat([x[1] for x in cached_batches])

    del cached_batches
    torch.cuda.empty_cache()

    if keep_gpu:
        # Put all cached data on GPU for faster optimization
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        cached_ress = cached_ress.to(device)

    # build optimizer and lr scheduler
    optimizer = torch.optim.SGD([module.weight], lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iter, eta_min=0.)

    model.set_spike_state(use_spike=True)
    # start optimization
    for i in range(optimize_iter):
        idx = torch.randperm(num_cali_samples)[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_res = cached_ress[idx].to(device)
        optimizer.zero_grad()
        module.zero_grad()

        snn_out = module.fwd_func(cur_inp, module.weight, module.bias, **module.fwd_kwargs) + cur_res
        snn_out = torch.clamp(snn_out / module.threshold * model.T, min=0, max=model.T)
        snn_out = floor_ste(snn_out) * module.threshold / model.T
        err = loss_func(snn_out, cur_out)
        err.backward()
        if dist_avg:
            allreduce(module.weight.grad)
        optimizer.step()
        scheduler.step()

        if i % 300 == 0:
            print('Iteration {}\tLoss: {},'.format(i, err.item()))

    print("\n")
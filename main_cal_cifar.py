import argparse
import os
import random

import numpy as np
import torch

from main_train_cifar import build_data
from models.calibration import bias_corr_model
from models.CIFAR.models.resnet import res_specials
from models.CIFAR.models.resnet import resnet20 as resnet20_cifar
from models.CIFAR.models.resnet import resnet32 as resnet32_cifar
from models.CIFAR.models.vgg import VGG
from models.fold_bn import search_fold_and_remove_bn
from models.spiking_layer import SpikeModel, get_maximum_activation


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def validate_model(test_loader, ann):
    correct = 0
    total = 0
    ann.eval()
    device = next(ann.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    return 100 * correct / total


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res20'])
    parser.add_argument('--dpath', default='./datasets', type=str, help='dataset directory')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')

    args = parser.parse_args()
    results_list = []
    use_bn = args.usebn

    if args.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we run the experiments for 5 times, with different random seeds
    for i in range(5):

        seed_all(seed=args.seed + i)
        sim_length = 32

        use_cifar10 = args.dataset == 'CIFAR10'

        train_loader, test_loader = build_data(dpath=args.dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True)

        if args.arch == 'VGG16':
            ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
        elif args.arch == 'res20':
            ann = resnet20_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
        elif args.arch == 'res32':
            ann = resnet32_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
        else:
            raise NotImplementedError

        load_path = 'raw/' + args.dataset + '/' + args.arch + '_wBN_wd5e4_state_dict.pth' if use_bn else \
            'raw/' + args.dataset + '/' + args.arch + '_woBN_wd1e4_state_dict.pth'
        if args.model != '':
            load_path = args.model

        state_dict = torch.load(load_path, map_location=device)
        ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.cuda()

        snn = SpikeModel(model=ann, sim_length=sim_length, specials=res_specials)
        snn.cuda()

        mse = False if args.calib == 'none' else True
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None,
                               sim_length=sim_length, channel_wise=False)

        if args.calib == 'light':
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
        if args.calib == 'advanced':
            # For CIFAR10/100 dataset, calibrating the potential only achieves best results.
            # weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=True)

        snn.set_spike_state(use_spike=True)
        results_list.append(validate_model(test_loader, snn))

    a = np.array([results_list])
    print(a.mean(), a.std())

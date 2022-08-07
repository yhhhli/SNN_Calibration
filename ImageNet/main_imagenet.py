import torch
import torchvision
import os
import random
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from ImageNet.models.vgg import vgg16, vgg16_bn, vgg_specials
from ImageNet.models.resnet import resnet34_snn, res_spcials
from CIFAR.models.calibration import bias_corr_model, weights_cali_model
from CIFAR.models.fold_bn import search_fold_and_remove_bn
from CIFAR.models.spiking_layer import SpikeModel, get_maximum_activation
from distributed_utils import initialize, get_local_rank
from ImageNet.models.mobilenet import mobilenetv1


def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4,
                        dist_sample: bool = False):
    print('==> Using Pytorch Dataset')

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader


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

    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res34'])
    parser.add_argument('--dpath', required=True, type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')

    parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')

    # Initialize distributed envrionments
    # Note: If this doesn't work, you may use the method in official torch example:
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    try:
        initialize()
        initialized = True
        torch.cuda.set_device(get_local_rank())
    except:
        print('For some reason, your distributed environment is not initialized, this program may run on separate GPUs')
        initialized = False

    args = parser.parse_args()
    results_list = []
    use_bn = args.usebn

    # run one time imagenet experiment.
    for i in range(1):

        seed_all(seed=args.seed + i)
        sim_length = 32

        use_cifar10 = args.dataset == 'CIFAR10'

        train_loader, test_loader = build_imagenet_data(data_path=args.dpath, dist_sample=initialized)

        if args.arch == 'VGG16':
            ann = vgg16_bn(pretrained=True) if args.usebn else vgg16(pretrained=True)
        elif args.arch == 'res34':
            ann = resnet34_snn(pretrained=True, use_bn=args.usebn)
        elif args.arch == 'mobilenet':
            ann = mobilenetv1(pretrained=True)
        else:
            raise NotImplementedError

        search_fold_and_remove_bn(ann)
        ann.cuda()

        snn = SpikeModel(model=ann, sim_length=sim_length, specials=vgg_specials if args.arch =='VGG16' else res_spcials)
        snn.cuda()

        mse = False if args.calib == 'none' else True
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None,
                               sim_length=sim_length, channel_wise=False, dist_avg=initialized)

        # make sure dist_avg=True to synchronize the data in different GPUs, e.g. gradient and threshold
        # otherwise each gpu performs its own calibration

        if args.calib == 'light':
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False, dist_avg=initialized)
        if args.calib == 'advanced':
            weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5,
                               dist_avg=initialized)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=True, dist_avg=initialized)

        snn.set_spike_state(use_spike=True)
        results_list.append(validate_model(test_loader, snn))

    a = np.array([results_list])
    print(a.mean(), a.std())
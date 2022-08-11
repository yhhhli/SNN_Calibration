import argparse
import os
import platform
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from data.autoaugment import CIFAR10Policy, Cutout
from models.CIFAR.models.resnet import resnet20, resnet32
from models.CIFAR.models.vgg import VGG


def build_data(dpath='./datasets', batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10/'),
                                train=True, download=False, transform=transform_train)
        val_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10/'),
                              train=False, download=False, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                                 train=True, download=False, transform=transform_train)
        val_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                               train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res20', type=str, help='dataset name',
                        choices=['CIFARNet', 'VGG16', 'CNN2', 'res20', 'res20m', 'mobv1'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='initial learning_rate')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--m', default='normal', type=str, help='calibration type',
                        choices=['normal', 'light', 'advanced'])
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    use_cifar10 = args.dataset == 'CIFAR10'

    num_workers = args.workers if not platform.system().lower() == 'windows' else 0
    train_loader, test_loader = build_data(batch_size, cutout=True, workers=num_workers, use_cifar10=use_cifar10, auto_aug=True)
    best_acc = 0
    best_epoch = 0
    use_bn = args.usebn
    model_save_name = 'raw/' + args.dataset + '/' + args.arch + '_wBN_wd5e4_state_dict.pth' if use_bn else \
                      'raw/' + args.dataset + '/' + args.arch + '_woBN_wd1e4_state_dict.pth'

    if args.arch == 'CNN2':
        raise NotImplementedError
    elif args.arch == 'VGG16':
        ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    elif args.arch == 'res20':
        ann = resnet20(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'res32':
        ann = resnet32(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
    else:
        raise NotImplementedError

    num_epochs = args.epochs
    start_epoch = 0
    ann.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    # build optimizer
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) if use_bn else \
                torch.optim.SGD(ann.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

    if args.resume:
        ckpt = torch.load(model_save_name)
        if ckpt.get('best_acc'):
            best_acc = ckpt['best_acc']

        if ckpt.get('epoch'):
            start_epoch = ckpt['epoch']

        if ckpt.get('model'):
            ann.load_state_dict(ckpt['model'])

        if ckpt.get('loss') and ckpt.get('optimizer') and ckpt.get('scheduler'):
            criterion = ckpt['loss']
            optimizer = ckpt['optimizer']
            scheduler = ckpt['scheduler']
   

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        pbar = tqdm(enumerate(train_loader))
        for i, (images, labels) in pbar:
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)
            outputs = ann(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                pbar.set_description('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Time elapsed: %.3fs'
                        % (epoch + 1, num_epochs, i + 1, len(train_loader) // batch_size, running_loss/100, time.time() - start_time))
                running_loss = 0

        scheduler.step()
        correct = 0
        total = 0

        # start testing
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = ann(inputs)
                loss = criterion(outputs, targets)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets.cpu()).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1

            if not os.path.exists(os.path.dirname(model_save_name)):
                os.makedirs(os.path.dirname(model_save_name))

            file, ext = os.path.splitext(model_save_name)
            model_best_name = file + '_best' + ext
            torch.save(ann.state_dict(), model_best_name)

        torch.save({'model': ann.state_dict(), 'epoch': epoch,
                    'best_acc': best_acc, 'loss': criterion, 'optimizer': optimizer, 'scheduler': scheduler},
                   model_save_name)
        print(f'Iters: {epoch + 1} best_acc is: {best_acc} find in epoch: {best_epoch}\n')


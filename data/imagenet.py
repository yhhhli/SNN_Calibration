import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageNetDataset
from .sampler import DistributedGivenIterationSampler, DistributedEpochSampler, DistributedSampler

try:
    import linklink.dali as link_dali
except ModuleNotFoundError:
    print('import linklink.dali failed, linklink version should >= 0.2.0')


class imgnet_CFG(object):

    def __init__(self, batchsize, workers, size):
        self.use_dali = False
        self.use_dali_v2 = False
        self.read_from = 'mc'
        self.batch_size = batchsize
        self.workers = workers
        self.pin_memory = True
        self.epoch_wise_shuffle = False

        self.input_size = size
        self.test_resize = size + 32

        self.max_iter = 2000
        self.last_iter = 0

        self.val_root = '/mnt/lustre/share/images/val'
        self.val_meta = '/mnt/lustre/share/images/meta/val.txt'
        self.train_root = '/mnt/lustre/share/images/train'
        self.train_meta = '/mnt/lustre/share/images/meta/train.txt'


def make_imagenet_train_data(config):
    """
    config fields:
        use_dali: bool
        read_from: 'fake', 'mc', 'ceph', 'fs'
        batch_size: int
        dali_workers: int
        workers: int
        pin_memory: bool
        epoch_wise_shuffle: bool
        input_size: int
        test_resize: int
        augmentation:
            rotation: float
            colorjitter: bool
        train_root: str
        train_meta: str
        val_root: str
        val_meta: str

        max_iter: int
        last_iter: int
    """

    if config.use_dali:
        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            read_from=config.read_from)

        pipeline = ImageNetTrainPipe(config.batch_size,
                                     config.dali_workers,
                                     torch.cuda.current_device(),
                                     config.input_size,
                                     colorjitter=config.augmentation.colorjitter)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        torch_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler,
            collate_fn=dali_default_collate)

        loader = DaliDataLoader(pipeline, dataloader=torch_loader)

    elif config.use_dali_v2:
        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        pipeline = ImageNetTrainPipeV2(config.train_root,
                                       config.train_meta,
                                       sampler,
                                       config.input_size,
                                       colorjitter=config.augmentation.colorjitter)

        loader = link_dali.DataLoader(pipeline, config.batch_size, len(sampler),
                                      config.dali_workers,
                                      last_iter=config.last_iter)

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        aug = [transforms.RandomResizedCrop(config.input_size),
               transforms.RandomHorizontalFlip()]

        aug.append(transforms.ColorJitter(*[0.2, 0.2, 0.2, 0.1]))
        aug.append(transforms.ToTensor())
        aug.append(normalize)

        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            transforms.Compose(aug),
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            print('use normal sampler')
            sampler = DistributedSampler(
                dataset=dataset)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=sampler)

    return loader


def make_imagenet_val_data(config):

    if config.use_dali:
        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            read_from=config.read_from)

        pipeline = ImageNetValPipe(config.batch_size,
                                   config.dali_workers,
                                   torch.cuda.current_device(),
                                   config.input_size,
                                   config.test_resize)

        sampler = DistributedSampler(dataset, round_up=False)

        torch_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler,
            collate_fn=dali_default_collate)

        loader = DaliDataLoader(pipeline, dataloader=torch_loader)

    elif config.use_dali_v2:

        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            read_from=config.read_from)

        sampler = DistributedSampler(dataset, round_up=False)

        pipeline = ImageNetValPipeV2(config.val_root,
                                     config.val_meta,
                                     sampler,
                                     config.input_size,
                                     config.test_resize)

        loader = link_dali.DataLoader(
            pipeline, config.batch_size, len(sampler), config.dali_workers)

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            transforms.Compose([
                transforms.Resize(config.test_resize),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                normalize,
            ]),
            read_from=config.read_from)

        sampler = DistributedSampler(dataset, round_up=False)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler)

    return loader

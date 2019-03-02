"""
Create train, valid, test iterators for CIFAR-10 [1].
[1]: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
"""

import torch
import numpy as np
from core.visualize import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging

class CIFAR10_train:

    def __init__(self,
                data_dir,
                batch_size,
                augment,
                random_seed,
                split_size=0.1,
                shuffle=True,
                show_sample=False,
                num_workers=4,
                pin_memory=False):
        """
        Utility function for loading and returning train and valid
        multi-process iterators over the CIFAR-10 dataset. A sample
        9x9 grid of the images can be optionally displayed.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - augment: whether to apply the data augmentation scheme
        mentioned in the paper. Only applied on the train split. TODO: Read this paper
        - random_seed: fix seed for reproducibility.
        - split_size: percentage split of the training set used for
        the validation set. Should be a float in the range [0, 1].
        - shuffle: whether to shuffle the train/validation indices.
        - show_sample: plot 9x9 sample grid of the dataset.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        """
        if (split_size < 0) or (split_size > 1):
            error_msg = "Expecting split size between [0,1] but received {}".format(split_size)
            logging.error(error_msg)
            raise ValueError(error_msg)

        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        # define transforms
        valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ])
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(split_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        self.classes = [
                        'airplane',
                        'automobile',
                        'bird',
                        'cat',
                        'deer',
                        'dog',
                        'frog',
                        'horse',
                        'ship',
                        'truck']
        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=9, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy().transpose([0, 2, 3, 1])
            plot_images(X, labels)

    def get_dataset_iterators(self):
        return self.train_loader, self.valid_loader

class CIFAR10_test:

    def __init__(self, data_dir,
                        batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False):
        """
        Utility function for loading and returning a multi-process
        test iterator over the CIFAR-10 dataset.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

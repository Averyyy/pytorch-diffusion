import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
# from torchvision import transforms

import glob
import random
import os

from PIL import Image
import torchvision.transforms as transforms


class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST"):

        transform = transforms.Compose([transforms.ToTensor()])

        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        train_dataset = datasets[dataset](
            "./data", download=True, train=train, transform=transform
        )
        # train_dataset = datasets[dataset](
        #     "./data", download=True, train=train
        # )

        self.dataset_len = len(train_dataset.data)

        if dataset == "MNIST" or dataset == "Fashion":
            pad = transforms.Pad(2)
            data = pad(train_dataset.data)
            data = data.unsqueeze(3)
            self.depth = 1
            self.size = 32
        elif dataset == "CIFAR":
            data = torch.Tensor(train_dataset.data)
            self.depth = 3
            self.size = 32
        self.input_seq = ((data / 255.0) * 2.0) - 1.0
        print("input_seq:", self.input_seq.shape)
        self.input_seq = self.input_seq.moveaxis(3, 1)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item]


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):

        self.transform = transforms.Compose(transforms_)
        self.depth = 3
        self.size = 256

        self.files = sorted(
            glob.glob(os.path.join(root, '%s' % mode) + '/*.*'))
        # self.input_seq = torch.Tensor(self.transform(
        #     Image.open(self.files[0]))).unsqueeze(0)
        # for i in range(1, len(self.files)):
        #     # print(i)
        #     self.input_seq = torch.cat((self.input_seq, torch.Tensor(
        #         self.transform(Image.open(self.files[i]))).unsqueeze(0)), dim=0)
        # self.input_seq = self.input_seq.moveaxis(1, 3)

    def __getitem__(self, index):

        img = self.transform(Image.open(self.files[index % len(self.files)]))
        print("img shape: ", img.shape)
        return img

    def __len__(self):
        return len(self.files)

import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

class Binarize(torch.nn.Module):
    def forward(self, img):
        img = torch.where(img == 0.0, img, 1.0)
        return img

def get_MNIST():
    input_size = 28
    input_channels = 1

    transform = transforms.Compose([transforms.ToTensor(), Binarize()])
    MNIST = datasets.MNIST(root="Datasets", train=True, transform=transform, download=True)

    return MNIST, input_size, input_channels

def get_CIFAR10():
    input_size = 32
    input_channels = 3

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), transforms.CenterCrop(input_size)])
    CIFAR10 = datasets.CIFAR10(root="Datasets", train=True, transform=transform, download=True)

    return CIFAR10, input_size, input_channels

def get_CelebA():
    raise NotImplementedError

    input_size = 32
    input_channels = 3

    #TODO check if this is the right transforms for CelebA
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(input_size), transforms.CenterCrop(input_size), transforms.Normalize((0.0,), (1.0,))])
    #TODO figure out how to get gdown to work
    CIFAR10 = datasets.CelebA(root="Datasets", download=True)

    return CIFAR10, input_size, input_channels

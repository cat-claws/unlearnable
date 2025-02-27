import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms

import io
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from . import datasets

transforms_3 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])

transforms_1 = transforms.ToTensor()

def auto_sets(name):
    if name == 'MNIST':
        train_set = datasets.auto_set('MNIST', download=True, train = True, transform=transforms_1)
        val_set = datasets.auto_set('MNIST', download=False, train = False, transform=transforms_1)
        channel = 1
    elif name == 'CIFAR10':
        train_set = datasets.auto_set('CIFAR10', download=True, train = True, transform=transforms_3)
        val_set = datasets.auto_set('CIFAR10', download=False, train = False, transform=transforms_1)
        channel = 3
    elif name == 'SVHN':
        train_set =  datasets.auto_set('SVHN', download=True, split = 'train', transform=transforms_1)
        val_set =  datasets.auto_set('SVHN', download=True, split = 'test', transform=transforms_1)
        channel = 3
    return train_set, val_set, channel


# batch process sampled inputs
def forward_micro(net, xs, microbatch_size):
    d0, d1, _, _, _ = xs.shape
    num_inputs = d0 * d1
    outputs = []
    for k in range(int(np.ceil(num_inputs / microbatch_size))):
        outputs.append(net(xs.view(num_inputs, *xs.shape[2:])[k * microbatch_size: (k + 1) * microbatch_size]))
    outputs = torch.cat(outputs, dim = 0).view(d0, d1, -1)
    return outputs

def xavier_init(net):
    if isinstance(net, nn.Conv2d):
        nn.init.xavier_normal_(net.weight, gain=nn.init.calculate_gain('relu'))
        if net.bias is not None:
            nn.init.zeros_(net.bias)
    elif isinstance(net, nn.Linear):
        nn.init.xavier_normal_(net.weight)
        if net.bias is not None:
            nn.init.zeros_(net.bias)
    # elif isinstance(net, nn.BatchNorm2d):
    #     net.weight.data.fill_(1)
    #     net.bias.data.zero_()


def kaiming_init(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            param.data.fill_(0)
        elif len(param.shape) < 2:
            param.data.fill_(1)
        else:
            nn.init.kaiming_normal_(param)

def plot_trend(mat, title, xticks, yticks, xlabel, ylabel):
    figure = plt.figure(figsize=(10, 8))
    plt.imshow(mat.T, interpolation='nearest', cmap=plt.cm.Blues, vmin=0., vmax=1.0)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(class_names))
    plt.xticks(np.arange(len(xticks)), xticks)#, rotation=45)
    plt.yticks(np.arange(len(yticks)), np.around(yticks, decimals=2))

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(mat.astype('float'), decimals=4)

    # Use white text if squares are dark; otherwise black.
    threshold = mat.max() / 2.
    for i, j in itertools.product(range(len(xticks)), range(len(yticks))):
        color = "white" if mat[i, j] > threshold else "black"
        plt.text(i, j, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    return figure

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    decoded = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)
    image = torch.from_numpy(decoded).permute(2, 0, 1)
    image = torch.cat((image[:3].flip(0), image[3:]))
    return image

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

from nets.mnist import ConvNet
from utils import experiment, iterate
from utils.sampling import sample_uniform_linf_with_clamp, sample_uniform_l2

import numpy as np
from ensemble import SampleEnsemble

m = ConvNet()
m = m.to('cuda' if torch.cuda.is_available() else 'cpu')


for eps_attack in [0.5, 1, 2, 4, 8]:
    for ckpt in ['checkpoints/ConvNet_TRADES.pt', 'checkpoints/ConvNet.pt', 'checkpoints/ConvNet_CVaR.pt']:

        writer = SummaryWriter(comment=ckpt+str(eps_attack))
        m.load_state_dict({k:torch.load(ckpt)[k] for k in m.state_dict()})

        k = 0
        iterate.attack(m,
                iterate.mnist_step,
                iterate.mnist_attacked_step,
                device = 'cuda',
                val_set = experiment.val_set,
                batch_size = 1000,
                epoch = k,
                writer = writer,
                torchattack=torchattacks.PGD,
                eps=eps_attack,
                alpha=1/255,
                steps=60,
                random_start=False
            )
        k += 1

        ns_neighb = range(1, 4)
        epsilons = np.linspace(0.1, 0.4, 4)
        original_accuracy = np.empty((len(ns_neighb), len(epsilons)))
        attacked_accuracy = np.empty((len(ns_neighb), len(epsilons)))

        for i, n_neighb in enumerate(ns_neighb):
            print(n_neighb)
            for j, eps in enumerate(epsilons):
                m_ = SampleEnsemble(m, sampling = sample_uniform_l2, n_neighb = n_neighb, eps = eps, batch_size = 4000)
                m_.eval()

                outputs, outputs_ = iterate.attack(m_,
                    iterate.mnist_step,
                    iterate.mnist_attacked_step,
                    device = 'cuda',
                    val_set = experiment.val_set,
                    batch_size = 500,
                    epoch = k,
                    writer = writer,
                    torchattack=torchattacks.PGD,
                    eps=eps_attack,
                    alpha=1/255,
                    steps=60,
                    random_start=False
                )
                k += 1

                original_acc, attacked_acc = outputs['correct'] / len(experiment.val_set), outputs_['correct'] / len(experiment.val_set)
                original_accuracy[i, j] = original_acc
                attacked_accuracy[i, j] = attacked_acc

        writer.add_image('original accuracy', experiment.plot_to_image(experiment.plot_trend(original_accuracy, 'original accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta')), 1)
        writer.add_image('attacked accuracy', experiment.plot_to_image(experiment.plot_trend(attacked_accuracy, 'attacked accuracy', ns_neighb, epsilons, 'num of neighbours', 'size of delta')), 1)
        writer.flush()
        writer.close()
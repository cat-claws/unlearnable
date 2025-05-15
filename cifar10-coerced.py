import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step, build_optimizer, build_scheduler, save_hparams, attacked_classification_step

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from shift import shift_towards_nearest_other_class
# from utils import get_gpu_usage

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--scheduler', type=str)

parser.add_argument('--model-depth', type=int)
parser.add_argument('--model-widen_factor', type=int)
parser.add_argument('--model-drop_rate', type=float)
parser.add_argument('--model-layers', type=int, nargs='+')
parser.add_argument('--model-block', type=str)


parser.add_argument('--optimizer-lr', type=float)
parser.add_argument('--optimizer-momentum', type=float)
parser.add_argument('--optimizer-weight_decay', type=float)
parser.add_argument('--optimizer-nesterov', action='store_true')
parser.add_argument('--optimizer-betas', type=lambda s: tuple(map(float, s.split(','))))
parser.add_argument('--optimizer-eps', type=float)

parser.add_argument('--scheduler-T_max', type=int)
parser.add_argument('--scheduler-eta_min', type=float)
parser.add_argument('--scheduler-step_size', type=int)
parser.add_argument('--scheduler-gamma', type=float)

parser.add_argument('--dataset', type=str)
parser.add_argument('--path', type=str)

parser.add_argument('--extra_train', type=float)
parser.add_argument('--n_components', type=int)
parser.add_argument('--training_step', type=str)
parser.add_argument('--validation_step', type=str, default='classification_step')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--posion-eps', type=float)
parser.add_argument('--note', type=str)

parser.add_argument('--atk', type=str)
parser.add_argument('--atk-eps', type=float)
parser.add_argument('--atk-alpha', type=float)
parser.add_argument('--atk-steps', type=int)


config = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
print(config)

writer = SummaryWriter(comment = f"_{config['dataset']}_{config['model']}", flush_secs=10)
save_hparams(writer, config, metric_dict={'Epoch-correct/valid': 0})


model = torch.hub.load('cat-claws/nn', config['model'], pretrained= False, **{k[6:]: config.pop(k) for k in list(config) if k.startswith('model_')}).to(config['device'])



config.update({k: eval(v) for k, v in config.items() if k.endswith('_step')})
config['optimizer'] = build_optimizer(config, [p for p in model.parameters() if p.requires_grad])
config['scheduler'] = build_scheduler(config, config['optimizer'])

if 'atk' in config:
    from utils import build_atk
    config['atk'] = build_atk(config, model)

import torchvision

train_transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, padding=4),
    # torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.ColorJitter(                 # Randomly change brightness, contrast, saturation
    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    # ),
    # torchvision.transforms.RandomApply([                # Randomly apply Gaussian Blur
    #     torchvision.transforms.GaussianBlur(kernel_size=3)
    # ], p=0.2),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_set = torch.hub.load('cat-claws/datasets', 'CIFAR10', path = 'cat-claws/'+config['path'], name = config['dataset'], split='train', transform = train_transform)
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

for epoch in range(config['epochs']):
    # get_gpu_usage()
    if epoch > 0:
        train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

    validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

    # torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

# print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
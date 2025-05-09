import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

from torchiteration import train, validate, predict, classification_step, predict_classification_step

import numpy as np
import itertools
import os
import copy
from collections import OrderedDict


from sklearn.datasets import fetch_openml
from shift import shift_towards_nearest_other_class
from utils import generate_combinations, flatten_dict

param_grid = OrderedDict({
    'dataset': ['cifar10'],
    'epsilon': np.power(2, np.arange(8)) / 255,
    'batch_size': [16, 32, 64],
    'n_components': [2, 8, 32, 128, 256],
    'extra_train': np.hstack(([1e-4], np.linspace(0.1, 0.9, 9), [1 - 1e-4])),
    'training_step': ['classification_step'],
    'validation_step': ['classification_step'],
    'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
    'optimizer': {
        # 'Adadelta': {'lr': [1.0, 0.1]},
        # 'Adam': {'lr': [1e-3, 1e-4], 'betas': [(0.9, 0.999)]},
        # 'SGD': {'lr': [0.1], 'momentum': [0.9]},
        'AdamW':{'lr':[0.001], 'weight_decay': 5e-4}
    },
    'scheduler': {
        # 'StepLR': {'step_size': [50, 100], 'gamma': [0.95, 0.1]},
        # 'ExponentialLR': {'gamma': [0.95, 0.99]},
        # 'MultiStepLR': {'milestones': [(100, 150, 200)], 'gamma': [0.1]},
        'CosineAnnealingLR':{'T_max': 200},
    }
})

models = [
    lambda: torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False, in_channels = 3),
    lambda: torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=False),
    lambda: torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False),
]

param_combinations = generate_combinations(copy.deepcopy(param_grid))

cifar = fetch_openml('CIFAR_10', cache=True, as_frame=False)
X = cifar.data.astype(np.float32)
y = cifar.target.astype(np.int64)

val_size = int(1/6 * len(X))

os.makedirs("checkpoints", exist_ok=True)
completed_indices = {int(f.split("_")[1]) for f in os.listdir("checkpoints") if f.startswith(param_grid['dataset'][0]) and f.endswith(".pt")}
last_completed_index = max(completed_indices, default=-1)

for index, (config, model_fn) in enumerate(itertools.product(param_combinations, models)):
    if index <= last_completed_index:
        print(f"Skipping experiment {index}, already completed.")
        continue

    model = model_fn().to(config['device'])

    writer = SummaryWriter(log_dir=f"runs/{config['dataset']}_{index}", flush_secs=10)

    flat_hparams = {**flatten_dict(config), 'model': model._get_name()}

    config = copy.deepcopy(config)
    for k, v in config.items():
        if k.endswith('_step'):
            config[k] = eval(v)
        elif k == 'optimizer':
            config[k] = getattr(torch.optim, v)([p for p in model.parameters() if p.requires_grad], **config[k + '_config'])
            config['scheduler'] = getattr(torch.optim.lr_scheduler, config['scheduler'])(config[k], **config['scheduler_config'])

    extra_size = int(config['extra_train'] * (len(X) - val_size))
    train_indices, extra_indices, val_indices = torch.utils.data.random_split(range(len(X)), [len(X) - val_size - extra_size, extra_size, val_size])

    X_train, y_train = X[train_indices], y[train_indices]
    X_extra, y_extra = X[extra_indices], y[extra_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    try:
        shift = shift_towards_nearest_other_class(X_extra, y_extra, X_extra, y_extra, n_components=config['n_components'], epsilon=config['epsilon'])
    except Exception as e:
        flat_hparams['error'] = str(e)

    print(flat_hparams)
    exp, ssi, sei = hparams(hparam_dict = flat_hparams, metric_dict={'Epoch-correct/valid': 0})   
    writer.file_writer.add_summary(exp)                 
    writer.file_writer.add_summary(ssi)                 
    writer.file_writer.add_summary(sei)
    
    if 'error' in flat_hparams:
        writer.flush()
        writer.close()
        continue 

    X_private = np.clip(X_extra + shift, 0, 1)
    X_train = np.vstack((X_train, X_private)).reshape(-1, 3, 32, 32)
    y_train = np.hstack((y_train, y_extra))
    X_val = X_val.reshape(-1, 3, 32, 32)

    import torchvision.transforms as transforms
    from data import TransformTensorDataset


    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    # val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))

    # train_loader = torch.utils.data.DataLoader(train_set, num_workers=4, batch_size=config['batch_size'], shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, num_workers=4, batch_size=config['batch_size'])
    train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
    val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)


    train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)


    for epoch in range(300):
        train(model, train_loader=train_loader, epoch=epoch, writer=writer, **config)
        validate(model, val_loader=val_loader, epoch=epoch, writer=writer, **config)

        torch.save(model.state_dict(), f"checkpoints/{config['dataset']}_{index}_{epoch:03}.pt")

    writer.flush()
    writer.close()

import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step, build_optimizer, build_scheduler, save_hparams

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from shift import shift_towards_nearest_other_class

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--scheduler', type=str)

parser.add_argument('--model-depth', type=int)
parser.add_argument('--model-widen_factor', type=int)
parser.add_argument('--model-drop_rate', type=float)

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

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--extra_train', type=float)
parser.add_argument('--n_components', type=int)
parser.add_argument('--training_step', type=str, default='classification_step')
parser.add_argument('--validation_step', type=str, default='classification_step')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epsilon', type=float)

config = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
print(config)

writer = SummaryWriter(comment = f"_{config['dataset']}_{config['model']}", flush_secs=10)
save_hparams(writer, config, metric_dict={'Epoch-correct/valid': 0})

# model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=False).to(config['device']).to(config['device'])
# model = torch.hub.load('pytorch/vision:v0.10.0', , pretrained=False).to(config['device'])
# model = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False, in_channels = 3).to(config['device'])
# model = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False, in_channels = 3).to(config['device'])
model = torch.hub.load('cat-claws/nn', config['model'], pretrained= False, num_classes=10, depth=config['model_depth'], drop_rate=config['model_drop_rate'], widen_factor = config['model_widen_factor']).to(config['device'])

config.update({k: eval(v) for k, v in config.items() if k.endswith('_step')})
config['optimizer'] = build_optimizer(config, [p for p in model.parameters() if p.requires_grad])
config['scheduler'] = build_scheduler(config, config['optimizer'])

cifar = fetch_openml('CIFAR_10', cache=True, as_frame=False)
X = cifar.data.astype(np.uint8) / 255
y = cifar.target.astype(np.int64)

val_size = int(1/6 * len(X))
extra_size = int(config['extra_train'] * (len(X) - val_size))

train_indices, extra_indices, val_indices = torch.utils.data.random_split(range(len(X)), [len(X) - val_size - extra_size, extra_size, val_size])

X_train, y_train = X[train_indices], y[train_indices]
X_extra, y_extra = X[extra_indices], y[extra_indices]
X_val, y_val = X[val_indices], y[val_indices]

shift = shift_towards_nearest_other_class(X_extra, y_extra, X_extra, y_extra, n_components = config['n_components'], epsilon = config['epsilon'])
# X_private = X_extra
X_private = np.clip(X_extra + shift, 0, 1)
print(X_train, X_extra, X_private, shift)

X_train = np.vstack((X_train, X_private)).reshape(-1, 3, 32, 32)
y_train = np.hstack((y_train, y_extra))

X_val = X_val.reshape(-1, 3, 32, 32)

import torchvision.transforms as transforms
from data import TransformTensorDataset


train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)


# train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.uint8), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
# val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.uint8), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)
# train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
# val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))

train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

import subprocess

def get_gpu_usage():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True
    )
    usage = result.stdout.strip().split('\n')
    for idx, line in enumerate(usage):
        gpu_util, mem_used, mem_total = map(int, line.split(', '))
        print(f"GPU {idx}: {gpu_util}% used, {mem_used}MB / {mem_total}MB memory")

for epoch in range(200):
	if epoch > 0:
		get_gpu_usage()
		train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	# torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

# print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
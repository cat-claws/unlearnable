import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from shift import shift_towards_nearest_other_class


config = {
	'dataset':'cifar10',
	'extra_train':0.51,
    'epsilon':8/255,
    'n_components':2,
	'training_step':'classification_step',
	'batch_size':512,
	'scheduler': 'CosineAnnealingLR',
	'scheduler_config': {
		'T_max': 200,  # Decay at later stages
	},
	'optimizer': 'AdamW',
	'optimizer_config': {
		'lr': 0.001,  # Standard for CIFAR-10 with SGD
		# 'momentum': 0.9,
		'weight_decay': 5e-4,  # Commonly used for CIFAR-10
	},
	'device':'cuda',
	'validation_step':'classification_step',
}

model = torch.hub.load('cat-claws/nn', 'resnet18_cifar', pretrained=False).to(config['device'])
# model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=False).to(config['device']).to(config['device'])
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(config['device'])
# from torchvision.models import resnet18
# model = resnet18()
# import torch.nn as nn
# import torch.nn.functional as F

# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model.maxpool = nn.Identity()
# model.fc = nn.Linear(512, 10)
# model = model.to(config['device'])

writer = SummaryWriter(comment = f"_{config['dataset']}_{model._get_name()}_{config['training_step']}_{config['epsilon']}_{config['extra_train']}", flush_secs=10)

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = eval(v)
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

cifar = fetch_openml('CIFAR_10', cache=True, as_frame=False)
X = cifar.data.astype(np.uint8)
y = cifar.target.astype(np.int64)

val_size = int(1/6 * len(X))
print(val_size)
extra_size = int(config['extra_train'] * (len(X) - val_size))

train_indices, extra_indices, val_indices = torch.utils.data.random_split(range(len(X)), [len(X) - val_size - extra_size, extra_size, val_size])

X_train, y_train = X[train_indices], y[train_indices]
X_extra, y_extra = X[extra_indices], y[extra_indices]
X_val, y_val = X[val_indices], y[val_indices]

shift = shift_towards_nearest_other_class(X_extra, y_extra, X_extra, y_extra, n_components = config['n_components'], epsilon = config['epsilon'])
X_private = X_extra#np.clip(X_extra + shift, 0, 1)
print(X_val, X_extra, X_train, X_private, '=============================')

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

train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.uint8), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.uint8), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)


train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)


for epoch in range(200):
	# if epoch > 0:
	train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	# torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

# print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
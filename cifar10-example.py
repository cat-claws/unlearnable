import csv
import itertools

# ---------------------------
# BASE CONFIG
# ---------------------------

base_config = {
    "python": "python",
    "script": "train.py",
    "--model": "WideResNet",
    "--model-depth": "28",
    "--model-widen_factor": "10",
    "--model-drop_rate": "0.3",
    "--optimizer": "SGD",
    "--optimizer-lr": "0.1",
    "--optimizer-momentum": "0.9",
    "--optimizer-weight_decay": "0.0005",
    "--scheduler": "CosineAnnealingLR",
    "--scheduler-T_max": "200",
    "--batch_size": "128",
}

# ---------------------------
# VARIANTS
# ---------------------------

model_variants = [
    {"--model-depth": "34"},
    {"--model-drop_rate": "0.0"},
]

optimizer_variants = [
    {"--optimizer": "AdamW", "--optimizer-lr": "0.001", "--optimizer-weight_decay": "0.01"},
]

parser.add_argument('--model-depth', type=int)
parser.add_argument('--model-widen_factor', type=int)
parser.add_argument('--model-drop_rate', type=float)

variant_groups = [model_variants, optimizer_variants, scheduler_variants]

# ---------------------------
# Ablation & Pairwise Combo
# ---------------------------

def generate_ablation_configs(base, variant_groups):
    configs = []
    for group in variant_groups:
        for variant in group:
            cfg = base.copy()
            cfg.update(variant)
            configs.append(cfg)
    return configs

def generate_pairwise_combos(base, variant_groups):
    combos = []
    for g1, g2 in itertools.combinations(variant_groups, 2):
        for v1 in g1:
            for v2 in g2:
                cfg = base.copy()
                cfg.update(v1)
                cfg.update(v2)
                combos.append(cfg)
    return combos

# ---------------------------
# Save as .sh (TSV format)
# ---------------------------

# model = torch.hub.load('cat-claws/nn', 'resnet_cifar', pretrained= False, num_classes=10, blocks=14, bottleneck=False, in_channels = 3).to(config['device'])
# model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=False).to(config['device']).to(config['device'])
# model = torch.hub.load('pytorch/vision:v0.10.0', , pretrained=False).to(config['device'])
model = torch.hub.load('cat-claws/nn', config['model'], pretrained= False, num_classes=10, depth=28, drop_rate=0.3, widen_factor = 10).to(config['device'])
# from torchvision.models import resnet18
# model = resnet18()
# import torch.nn as nn
# import torch.nn.functional as F

# model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# model.maxpool = nn.Identity()
# model.fc = nn.Linear(512, 10)
# model = model.to(config['device'])

shift = shift_towards_nearest_other_class(X_extra, y_extra, X_extra, y_extra, n_components = config['n_components'], epsilon = config['epsilon'])
X_private = X_extra#np.clip(X_extra + shift, 0, 1)

X_train = np.vstack((X_train, X_private)).reshape(-1, 3, 32, 32)
y_train = np.hstack((y_train, y_extra))

X_val = X_val.reshape(-1, 3, 32, 32)

import torchvision.transforms as transforms
from data import TransformTensorDataset


# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
# ])

# test_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.uint8), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
# val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.uint8), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)
train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))

train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)


for epoch in range(200):
	if epoch > 0:
		train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	# torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

# print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()

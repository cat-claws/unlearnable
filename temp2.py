import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18

from data import TransformTensorDataset

def seed_everything(seed=42):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # cudnn.deterministic = True
    # cudnn.benchmark = False

seed_everything()

from sklearn.datasets import fetch_openml
cifar = fetch_openml('CIFAR_10', cache=True, as_frame=False)
X = cifar.data.astype(np.uint8).reshape(-1, 3, 32, 32)
y = cifar.target.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=10000)#, random_state=42)

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
print(X_train)
train_set = TransformTensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64), transform=train_transform)
val_set = TransformTensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64), transform=test_transform)


trainloader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
testloader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

# ------------------ Model ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet18()
# net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)#.to(config['device'])

net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net.fc = nn.Linear(512, 10)
net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# ------------------ Loss and Optimizer ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

# ------------------ Training Loop ------------------
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device):
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Train Epoch {epoch}: Loss: {train_loss:.3f} | Acc: {100.*correct/total:.3f}%')

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f'Test Epoch {epoch}: Acc: {acc:.3f}%')
    return acc

# ------------------ Run Training ------------------
best_acc = 0
for epoch in range(1, 201):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(net.state_dict(), 'best_cifar10.pth')
        print(f"✅ Best Acc updated to {best_acc:.3f}% at epoch {epoch}")

print(f"\n🏁 Final Best Accuracy: {best_acc:.3f}%")

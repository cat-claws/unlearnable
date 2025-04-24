import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms - No augmentation, just normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# Model - ResNet18 with modified final layer
model = resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

# Training loop
def train(epoch):
    model.train()
    running_loss = 0.0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
import torch.backends.cudnn as cudnn
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18

# ------------------ Reproducibility ------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

seed_everything()

# ------------------ Data Augmentation ------------------
class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask).expand_as(img)
        return img * mask

# ------------------ Transforms ------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    Cutout(n_holes=1, length=8)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# ------------------ Dataloaders ------------------
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# ------------------ Model ------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet18()
net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
net.maxpool = nn.Identity()
net.fc = nn.Linear(512, 10)
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

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
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch} - Loss: {running_loss / len(trainloader):.4f}")

# Test function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# Run training
best_acc = 0
for epoch in range(200):
    train(epoch)
    acc = test()
    scheduler.step()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f"Best Test Accuracy: {best_acc:.2f}%")
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

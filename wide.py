import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ===============================
# WideResNet Definition
# ===============================

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes and stride == 1)
        self.shortcut = (lambda x: x) if self.equal_in_out else nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.conv1(self.relu(self.bn1(x)))
        out = self.relu(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + self.shortcut(x)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes if i == 0 else out_planes,
                                out_planes, stride if i == 0 else 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size(3)).view(out.size(0), -1)
        return self.fc(out)

# ===============================
# Training Script
# ===============================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transform - no augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
model = WideResNet(depth=28, widen_factor=10, num_classes=10, drop_rate=0.3).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=200)

# from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts

# warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
# cosine = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
# scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# Training function
def train(epoch):
    model.train()
    running_loss = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Train Loss: {running_loss / len(trainloader):.4f}")

# Testing function
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# Main loop
best_acc = 0
for epoch in range(305):
    train(epoch)
    acc = test()
    scheduler.step()

    if acc > best_acc:
        best_acc = acc
        # torch.save(model.state_dict(), 'best_wrn28-10.pth')

print(f'🎯 Best Test Accuracy: {best_acc:.2f}%')

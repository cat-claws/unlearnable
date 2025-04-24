import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# -----------------------------
# QuickNet Definition
# -----------------------------
class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class QuickNet(nn.Module):
    """
    QuickNet: entry stem -> separable-conv blocks with PReLU -> global avg pool -> classifier
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Entry stem: 5x5 conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        # Blocks: separable conv -> BN -> PReLU -> MaxPool -> Dropout
        channels = [64, 128, 256, 512, 1024]
        self.blocks = nn.ModuleList()
        in_ch = 64
        for out_ch in channels:
            blk = nn.Sequential(
                SeparableConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(out_ch),
                nn.MaxPool2d(2),
                nn.Dropout(0.5)
            )
            self.blocks.append(blk)
            in_ch = out_ch
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


# -----------------------------
# Training and Evaluation
# -----------------------------
def train_one_epoch(epoch, model, criterion, optimizer, dataloader, device, log_interval=100):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % log_interval == 0 or batch_idx == len(dataloader):
            print(f"Epoch {epoch} [{batch_idx * len(inputs)}/{len(dataloader.dataset)}]  "
                  f"Avg Loss: {running_loss / batch_idx:.4f}")

def evaluate(model, criterion, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    print(f"Test  Avg Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2f}%")
    return accuracy


# -----------------------------
# Main Script
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="QuickNet Training on CIFAR-10 (no augment)")
    parser.add_argument('--data-dir',    default='./data',      help='path to CIFAR-10 data')
    parser.add_argument('--batch-size',  type=int,   default=128, help='training batch size')
    parser.add_argument('--test-batch',  type=int,   default=100, help='test batch size')
    parser.add_argument('--epochs',      type=int,   default=200, help='number of epochs')
    parser.add_argument('--lr',          type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum',    type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay',type=float, default=5e-4, help='weight decay (L2)')
    parser.add_argument('--step-size',   type=int,   default=60,  help='LR step size (epochs)')
    parser.add_argument('--gamma',       type=float, default=0.2, help='LR decay factor')
    parser.add_argument('--checkpoint',  default='quicknet_ckpt.pth', help='checkpoint file')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data transforms (no augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # Datasets and loaders
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.test_batch,
                            shuffle=False, num_workers=4)

    # Model, loss, optimizer, scheduler
    model = QuickNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=args.step_size,
                                    gamma=args.gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(epoch, model, criterion, optimizer, trainloader, device)
        acc = evaluate(model, criterion, testloader, device)
        scheduler.step()

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_acc': best_acc,
                'optimizer_state': optimizer.state_dict(),
            }, args.checkpoint)
            print(f"Checkpoint saved at epoch {epoch} with acc {best_acc:.2f}%\n")

    print(f"Training complete. Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()


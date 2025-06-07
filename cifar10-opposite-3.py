import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm


import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(123)

model = torch.hub.load('cat-claws/nn', 'resnet_cifar', block='', layers=[2, 2, 2, 2], num_classes=10)
model.cuda()

from hardcoded_transforms import transforms
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms('cifar10_T'))
train_set = torch.hub.load('cat-claws/datasets', 'CIFAR10', path = 'cat-claws/poison', name = 'cifar10', split='train', transform = transforms('cifar10_T'), indexed=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms('cifar10_T'))
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

X = torch.stack([d[0].clone() for d in train_set]).cuda()
X_ = X.clone().requires_grad_(True)
# optimizer_X = torch.optim.AdamW([X_], lr=0.001, weight_decay=1e-2)
optimizer_X = torch.optim.SGD([X_], lr=0.1, weight_decay=5e-4, momentum=0.9)

def evaluate(model, loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    return loss / len(loader), acc

epsilon = 8 / 255

import torch
import torch.nn.functional as F

def grad_similarity(grads_a, grads_b, mode, reduction='mean'):
    sims = {
        'dot': lambda g1, g2: (g1 * g2).sum(),
        'l2_squared': lambda g1, g2: ((g1 - g2) ** 2).sum(),
        'l2': lambda g1, g2: (g1 - g2).norm(),
        'cosine': lambda g1, g2: F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0),
    }
    vals = [sims[mode](g1, g2) for g1, g2 in zip(grads_a, grads_b)]
    return sum(vals) if reduction == 'sum' else torch.stack(vals).mean()
     

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 200
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    optimizer_X.zero_grad()
    X_.grad = torch.zeros_like(X_)

    for batch_idx, (x, labels, indices) in enumerate(train_loader):
        x, labels = x.cuda(), labels.cuda()

        x_ = X_[indices]

        model.train()

        # Compute df/dy at x (reference gradients)
        clean_grad = [g.detach() for g in torch.autograd.grad(criterion(model(x), labels), model.parameters(), create_graph=False)]

        loss = criterion(model(x_), labels)                        
        
        poison_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        
        X_.grad[indices] = torch.autograd.grad(grad_similarity(clean_grad, poison_grad, 'dot', 'sum'), x_, retain_graph=True)[0]
        
        
        optimizer.step()

        with torch.no_grad():
            perturbation = torch.clamp(X_ - X, min=-epsilon, max=epsilon)
            X_.data = X + perturbation
            X_.data.clamp_(0.0, 1.0)
            
            print(f"[Epoch {epoch+1} | Batch {batch_idx:03d} ] "
                f"loss = {loss.item():.4f} "
                f"L2 = {grad_similarity(clean_grad, poison_grad, 'l2').item():.4f}, Cosine = {grad_similarity(clean_grad, poison_grad, 'cosine').item():.4f}")

    optimizer_X.step()

    scheduler.step()

    test_loss, test_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}] "
        f"Test Acc: {test_acc:.2f}% ")

from upload_utils import upload_tensor_dataset_to_hub

upload_tensor_dataset_to_hub(
    x_tensor=X_.detach().cpu(),
    dataset_repo="trial",
    config_name="resnet18-ad2-3-9sgd-smallbatch",
    private=False,
    token = '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'
)


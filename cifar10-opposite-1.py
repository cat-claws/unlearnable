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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms('cifar10_T'))
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

X_ = torch.stack([d[0].clone() for d in train_set])

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

def grad_loss(grad_1, grad_2, mode = 'dot', reduction = 'sum'): # Options: 'dot', 'l2', 'cosine'
    if mode == 'dot' and reduction == 'sum':
        return sum((g2 * g1).sum() for g2, g1 in zip(grad_1, grad_2))
    elif mode == 'l2' and reduction == 'sum':
        return sum(((g2 + g1) ** 2).sum() for g2, g1 in zip(grad_1, grad_2))
    elif mode == 'cosine' and reduction == 'sum':
        return sum(
            F.cosine_similarity(g2.flatten(), g1.flatten(), dim=0)
            for g2, g1 in zip(grad_1, grad_2)
        )
    elif mode == 'l2' and reduction == 'mean':
        return torch.stack([(g2 + g1).norm() for g2, g1 in zip(grad_1, grad_2)]).mean()
    elif mode == 'dot' and reduction == 'mean':
        return torch.stack([(g2 * g1).sum() for g2, g1 in zip(grad_1, grad_2)]).mean()
    elif mode == 'cosine' and reduction == 'mean':
        return torch.stack([F.cosine_similarity(g2.flatten(), g1.flatten(), dim=0) for g2, g1 in zip(grad_1, grad_2)]).mean()
                

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):    

    for batch_idx, (x, labels, indices) in enumerate(tqdm(train_loader)):
        x, labels = x.cuda(), labels.cuda()

        x_ = X_[indices].detach().cuda().requires_grad_(True)
        optimizer_x = torch.optim.AdamW([x_], lr=0.01, weight_decay=1e-4)

        model.train()
        for step in range(5):

            # Compute df/dy at x (reference gradients)
            clean_grad = [g.detach() for g in torch.autograd.grad(criterion(model(x), labels), model.parameters(), create_graph=False)]

            loss = criterion(model(x_), labels)                        
            
            poison_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            optimizer_x.zero_grad()
            x_.grad = torch.autograd.grad(grad_loss(clean_grad, poison_grad), x_, retain_graph=True)[0]
            optimizer_x.step()
            
            optimizer.step()

            with torch.no_grad():
                perturbation = torch.clamp(x_ - x, min=-epsilon, max=epsilon)
                x_.data = x + perturbation
                x_.data.clamp_(0.0, 1.0)

                
                print(f"[Epoch {epoch+1} | Batch {batch_idx:03d} | Step {step:02d}] "
                    f"loss = {loss.item():.4f} "
                    f"L2 = {grad_loss(clean_grad, poison_grad, 'l2', 'mean').item():.4f}, Dot = {grad_loss(clean_grad, poison_grad, 'dot', 'mean').item():.4f}, Cosine = {grad_loss(clean_grad, poison_grad, 'cosine', 'mean').item():.4f}")

        X_[indices] = x_.detach().cpu()

    scheduler.step()

    test_loss, test_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}] "
        f"Test Acc: {test_acc:.2f}% ")

    torch.save(X_, f'perturbed_epoch{epoch+1}.pt')
    print(f"✅ Saved perturbed x_ for epoch {epoch+1}.")


from upload_utils import upload_tensor_dataset_to_hub

upload_tensor_dataset_to_hub(
    x_tensor=X_,
    dataset_repo="trial",
    config_name="resnet18-ad2-3-1",
    private=False,
    token = ''
)

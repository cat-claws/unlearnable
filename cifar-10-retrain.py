# !pip install git+https://github.com/cestwc/sharpen/

# !pip uninstall bayeslap -y
# !pip install git+https://github.com/cat-claws/bayes-error-local-averaging/

# !pip uninstall torchadversarial -y
# !pip install git+https://github.com/cat-claws/torchadversarial/

# !rm -rf ~/.cache/torch

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

X = torch.from_numpy(train_set.data.transpose(0, 3, 1, 2)) / 255  # Shape: [50000, 3, 32, 32]
X = X.to(device)
y = torch.tensor(train_set.targets).to(device)  # Shape: (50000,)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


config = {
    'dataset': 'cifar10',

    'model': 'resnet_cifar',

    'training_step': 'classification_step',
    'validation_step': 'classification_step',

    'batch_size': 5120,
    'epochs': 20,

    'optimizer': 'SGD',
    'optimizer_momentum': 0.9,
    'optimizer_weight_decay': 5e-4,
    'optimizer_lr': 0.1,

    'scheduler': 'CosineAnnealingLR',
    'scheduler_T_max': 20,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)


from steps import feature_extraction_step
from torchiteration import predict, predict_classification_step#train, validate, predict, classification_step, predict_classification_step, build_optimizer, build_scheduler, save_hparams, attacked_classification_step


import bayeslap

be_calculator = lambda x, y, num=10, sigma=0.3, batch=32: bayeslap.BayesErrorRBF.apply(x, y, sigma, num, batch)
# be_calculator = lambda x, y, num=10, batch=32: bayeslap.BayesErrorLogistic.apply(x, y, num, batch)

# from sharpen import view

# view(X);

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    acc = 100. * correct / total
    return running_loss / len(loader), acc

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
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



num_epochs = 20
batch_size = 128 *16
learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
from tqdm import tqdm


# model = torch.hub.load(
#     'cat-claws/nn',
#     'wideresnet',
# )
# Data augmentation & normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

from torch.utils.data import TensorDataset

class TransformTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y

import torch
from torchadversarial import Attack


for x_ in Attack(torch.optim.AdamW, [X], steps = 200, foreach=False, maximize=True):

    with torch.no_grad():
        x_[0].copy_(x_[0].clamp(X - 8/255, X + 8/255).clamp(0, 1))  # ← modifies in-place, but autograd is off

    model = torch.hub.load(
        'cat-claws/nn',
        'resnet_cifar',
        block='',
        layers=[2, 2, 2, 2],
        num_classes=10,
    )

    model = model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    # num_epochs += 1

    p_train_loader =  torch.utils.data.DataLoader(TransformTensorDataset(x_[0].detach().clone(), y, transform=transform), batch_size=config['batch_size'], shuffle=False)

    # Training loop
    best_acc = 0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, p_train_loader, optimizer, criterion)
        # test_loss, test_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Acc: {train_acc:.2f}% ")
            # f"Test Acc: {test_acc:.2f}%")

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     torch.save(model.state_dict(), "best_resnet18_cifar10.pth")

    # print(f"Best Test Accuracy: {best_acc:.2f}%")
    # model.load_state_dict(torch.load("best_resnet18_cifar10.pth"))

    # normalized_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x_[0])

    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    
    normalized_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(x_[0])
    error = bayeslap.BayesErrorImageFeature.apply(normalized_images, y, model, be_calculator)
    print(f"{error.item():.3f}")

    error.backward()


    p_data = x_[0].detach().clamp(X - 8/255, X + 8/255).clamp(0, 1)


    (p_data - X).max()

    p_loader =  torch.utils.data.DataLoader(torch.utils.data.TensorDataset(p_data, y), batch_size=config['batch_size'], shuffle=False)
    outputs = predict(model, feature_extraction_step, val_loader = p_loader, **config)
    p_features = torch.tensor(outputs['predictions'])


    from upload_utils import upload_tensor_dataset_to_hub

    upload_tensor_dataset_to_hub(
        x_tensor=p_data,
        dataset_repo="trial",
        config_name="resnet18-retrain",
        private=False,
        token = ''
    )

# tsne_plot(train_features.numpy(), np.array(train_set.targets))

# hp_featuresf_WhoLgQeIRp_featuressnPAmCHdnuYp_featuresnahwwviUhTXgDO
# tsne_plot(p_features.numpy(), np.array(train_set.targets))

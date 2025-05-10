import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader

# ----- SimCLR Transform -----
class SimCLRTransform:
    def __init__(self):
        self.transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

# ----- Dataset Wrapper -----
class SimCLRDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.simclr_transform = SimCLRTransform()

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        x1, x2 = self.simclr_transform(x)
        return x1, x2

    def __len__(self):
        return len(self.base_dataset)

# ----- Projection Head -----
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----- SimCLR Model -----
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, proj_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.fc = nn.Identity()  # Remove classification head
        self.projector = ProjectionHead(512, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# ----- NT-Xent Loss -----
def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature

    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim.masked_fill_(mask, -9e15)

    pos_indices = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    positives = sim[torch.arange(2 * N), pos_indices]

    loss = -torch.log(torch.exp(positives) / torch.exp(sim).sum(dim=1))
    return loss.mean()

# ----- Training Function -----
def train_simclr(epochs=10, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,)
    train_dataset = SimCLRDataset(base_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    resnet = resnet18(pretrained=False)
    model = SimCLRModel(resnet).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x1, x2) in enumerate(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Run the training
train_simclr(20)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ----- Device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- Transforms -----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- CIFAR-10 Data -----
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# ----- Custom Pair Dataset for InfoNCE -----
class ContrastivePairDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets
        self.class_to_indices = self._build_index()

    def _build_index(self):
        c2i = {i: [] for i in range(10)}
        for idx, label in enumerate(self.targets):
            c2i[label].append(idx)
        return c2i

    def __getitem__(self, index):
        x_i, y_i = self.dataset[index]
        pos_index = random.choice(self.class_to_indices[y_i])
        while pos_index == index:
            pos_index = random.choice(self.class_to_indices[y_i])
        x_j, _ = self.dataset[pos_index]
        return x_i, x_j, y_i

    def __len__(self):
        return len(self.dataset)

# ----- Simple ConvNet -----
class EmbeddingNet(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(self.fc(x), p=2, dim=1)
        return x

# ----- InfoNCE Loss -----
def info_nce_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # 2N x D
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2N x 2N

    # Remove self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, -9e15)

    # Positive pairs: i<->j
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    pos_sim = sim_matrix[torch.arange(2 * batch_size), positives]

    loss = -torch.log(torch.exp(pos_sim / temperature) / torch.exp(sim_matrix / temperature).sum(dim=1))
    return loss.mean()

# ----- DataLoader -----
batch_size = 256
train_pair_dataset = ContrastivePairDataset(train_dataset)
train_loader = DataLoader(train_pair_dataset, batch_size=batch_size, shuffle=True)

# ----- Model Setup -----
model = EmbeddingNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----- Training -----
def train(model, loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_i, x_j, _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = info_nce_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

train(model, train_loader, 100)

# ----- Extract Embeddings -----
def extract_embeddings(model, dataset):
    model.eval()
    all_z = []
    all_y = []
    loader = DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model(x)
            all_z.append(z.cpu())
            all_y.extend(y)
    return torch.cat(all_z), torch.tensor(all_y)

# ----- k-NN Evaluation -----
def knn_eval(model, train_data, test_data, k=5):
    z_train, y_train = extract_embeddings(model, train_data)
    z_test, y_test = extract_embeddings(model, test_data)

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(z_train, y_train)
    preds = knn.predict(z_test)
    acc = accuracy_score(y_test, preds)
    print(f"k-NN Classification Accuracy (k={k}): {acc:.4f}")

knn_eval(model, train_dataset, test_dataset)
train(model, train_loader, 100)
knn_eval(model, train_dataset, test_dataset)
train(model, train_loader, 100)
knn_eval(model, train_dataset, test_dataset)
train(model, train_loader, 100)
knn_eval(model, train_dataset, test_dataset)
# train(model, train_loader, 100)

# ----- Optional: t-SNE Visualization -----
def tsne_plot(embeddings, labels):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings[:5000])
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels[:5000], cmap='tab10', s=5)
    plt.title("t-SNE of Learned Embeddings")
    plt.grid(True)
    plt.show()

# Run t-SNE
# z_test, y_test = extract_embeddings(model, test_dataset)
# tsne_plot(z_test.numpy(), y_test.numpy())

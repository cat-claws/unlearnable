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

# ----- Setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- Data Transforms -----
transform = transforms.Compose([
    transforms.ToTensor(),
])

# ----- Load CIFAR-10 -----
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# ----- Pairwise Dataset -----
class PairwiseCIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets
        self.class_to_indices = self._build_index_by_class()

    def _build_index_by_class(self):
        class_to_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(self.targets):
            class_to_indices[label].append(idx)
        return class_to_indices

    def __getitem__(self, index):
        x1, label1 = self.dataset[index]
        same_class = random.choice([True, False])
        if same_class:
            idx2 = random.choice(self.class_to_indices[label1])
            y = 1
        else:
            label2 = random.choice([l for l in range(10) if l != label1])
            idx2 = random.choice(self.class_to_indices[label2])
            y = 0
        x2, _ = self.dataset[idx2]
        return x1, x2, torch.tensor(float(y))

    def __len__(self):
        return len(self.dataset)

# ----- Embedding Network -----
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----- Similarity Model -----
class SimilarityModel(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        z1 = self.embedding_net(x1)
        z2 = self.embedding_net(x2)
        sim_score = torch.sum(z1 * z2, dim=1)
        return torch.sigmoid(sim_score)

# ----- Create Dataloaders -----
train_pairs = PairwiseCIFAR10(train_dataset)
test_pairs = PairwiseCIFAR10(test_dataset)

train_loader = DataLoader(train_pairs, batch_size=128, shuffle=True)
test_loader = DataLoader(test_pairs, batch_size=128, shuffle=False)

# ----- Model Setup -----
embedding_net = EmbeddingNet().to(device)
model = SimilarityModel(embedding_net).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----- Training -----
def train(model, loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x1, x2, labels in loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            preds = model(x1, x2)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

train(model, train_loader, 1000)

# ----- Evaluation (Pairwise Accuracy) -----
def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x1, x2, labels in dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            preds = model(x1, x2)
            predicted = (preds > threshold).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Pair Classification Accuracy: {correct / total:.4f}")

evaluate(model, test_loader)

# ----- Embedding Extraction -----
def extract_embeddings(model, dataset):
    model.eval()
    all_z = []
    all_y = []
    loader = DataLoader(dataset, batch_size=256)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.embedding_net(x)
            all_z.append(z.cpu())
            all_y.extend(y)
    return torch.cat(all_z), torch.tensor(all_y)

# ----- k-NN Evaluation -----
def knn_eval(train_dataset, test_dataset, model, k=5):
    print("Extracting embeddings...")
    z_train, y_train = extract_embeddings(model, train_dataset)
    z_test, y_test = extract_embeddings(model, test_dataset)

    print("Running k-NN classifier...")
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    knn.fit(z_train, y_train)
    preds = knn.predict(z_test)
    acc = accuracy_score(y_test, preds)
    print(f"k-NN Classification Accuracy (k={k}): {acc:.4f}")

knn_eval(train_dataset, test_dataset, model)

# ----- Optional: Visualize with t-SNE -----
def tsne_plot(embeddings, labels, title='t-SNE of CIFAR-10 embeddings'):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings[:5000])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels[:5000], cmap='tab10', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Run t-SNE (optional)
z_all, y_all = extract_embeddings(model, test_dataset)
tsne_plot(z_all.numpy(), y_all.numpy())

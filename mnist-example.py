import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

config = {
	'dataset':'mnist',
	'training_step':'classification_step',
	'batch_size':32,
	'optimizer':'Adadelta',
	'optimizer_config':{
		'lr':1,
	},
	'scheduler':'StepLR',
	'scheduler_config':{
		'step_size':2000,
		'gamma':1
	},
	'device':'cuda' if torch.cuda.is_available() else 'cpu',
	'validation_step':'classification_step',
	# 'sensitive_index':6,
}

# model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [13, 64, 32, 16, 8, 4], num_classes = 1).to(config['device'])
model = torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 16, 5), (16, 24, 5) ], linears = [24*4*4, 100]).to(config['device'])
# model = torch.hub.load('cat-claws/nn', 'exampleconvnet', in_channels = 1, pretrained = 'exampleconvnet_cbyC')

writer = SummaryWriter(comment = f"_{config['dataset']}_{model._get_name()}_{config['training_step']}", flush_secs=10)

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = eval(v)
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

import pandas as pd
import numpy as np

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

pca = PCA(n_components=15, whiten=False)
X_pca = pca.fit_transform(X)
print(np.isclose(pca.inverse_transform(X_pca), X, atol=1e-04))


X = X.reshape(-1, 1, 28, 28)

dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
train_set, val_set = torch.utils.data.random_split(dataset, [int(len(X)*0.7), len(X)-int(len(X)*0.7)])

X_train, y_train = train_set.dataset.tensors[0][train_set.indices], train_set.dataset.tensors[1][train_set.indices]
X_train_pca = pca.fit_transform(X_train.reshape(-1, 784))


# class_means = {}
# for label in range(10):  # MNIST has 10 classes (0-9)
#     class_means[label] = X_train[y_train == label].mean(dim=0)
class_means_pca = {label: X_train_pca[y_train == label].mean(axis=0) for label in range(10)}


# Function to shift each sample towards the nearest other class center
def shift_towards_nearest_other_class(X_train, y_train, class_means, alpha):
    X_train_shifted = X_train.clone()
    for i in range(len(X_train)):
        current_class = y_train[i].item()
        current_sample = X_train[i]

        # Find the nearest other class centroid
        nearest_class = min(
            (c for c in class_means if c != current_class),
            key=lambda c: torch.norm(class_means[c] - current_sample)
        )
        nearest_center = class_means[nearest_class]

        # Shift towards the nearest class center
        X_train_shifted[i] = (1 - alpha) * current_sample + alpha * nearest_center

    return X_train_shifted

def shift_towards_nearest_other_class(X_pca, y, class_means, alpha):
    X_shifted = X_pca.copy()
    for i in range(len(X_pca)):
        current_class = y[i]
        current_sample = X_pca[i]

        # Find nearest other class centroid
        nearest_class = min(
            (c for c in class_means if c != current_class),
            key=lambda c: np.linalg.norm(class_means[c] - current_sample)
        )
        nearest_center = class_means[nearest_class]

        # Shift towards nearest centroid
        X_shifted[i] = (1 - alpha) * current_sample + alpha * nearest_center

    return X_shifted

# Apply transformation only to training data
# X_train_shifted = shift_towards_nearest_other_class(X_train, y_train, class_means, alpha = 0.3)
X_train_pca_shifted = shift_towards_nearest_other_class(X_train_pca, y_train, class_means_pca, alpha = 0.3)
X_train_shifted = np.clip(pca.inverse_transform(X_train_pca_shifted), 0, 1).reshape(-1, 1, 28, 28)

print(X_train_shifted.shape, y_train.shape)

# Create new train dataset with shifted training data
train_dataset_shifted = torch.utils.data.TensorDataset(torch.tensor(X_train_shifted, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))

train_loader = torch.utils.data.DataLoader(train_dataset_shifted, num_workers = 4, batch_size = config['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, num_workers = 4, batch_size = config['batch_size'])


for epoch in range(100):
	if epoch > 0:
		train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

	validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

	torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
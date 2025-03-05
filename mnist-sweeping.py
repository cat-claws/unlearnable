import torch
from torch.utils.tensorboard import SummaryWriter
from torchiteration import train, validate, predict, classification_step, predict_classification_step

import numpy as np
import itertools
import os
from sklearn.datasets import fetch_openml
from shift import shift_towards_nearest_other_class

# Define hyperparameter grid
param_grid = {
    'epsilon': [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64],
    'n_components': [2, 3, 5],
    'optimizer': {
        'Adadelta': {'lr': [1.0, 0.1]},
        'Adam': {'lr': [1e-3, 1e-4], 'betas': [(0.9, 0.999)]},
        'SGD': {'lr': [0.1, 0.01], 'momentum': [0.9]},
    },
}

# Expand optimizer grid separately
optim_combinations = []
for optim, params in param_grid['optimizer'].items():
    param_keys, param_values = zip(*params.items())
    for values in itertools.product(*param_values):
        optim_combinations.append({'optimizer': optim, 'optimizer_config': dict(zip(param_keys, values))})

# Generate all non-optimizer parameter combinations
non_optim_keys = [k for k in param_grid if k != 'optimizer']
non_optim_values = [param_grid[k] for k in non_optim_keys]
non_optim_combinations = list(itertools.product(*non_optim_values))

# Combine both grids and index them
param_combinations = []
for i, (non_optim_values, optim_dict) in enumerate(itertools.product(non_optim_combinations, optim_combinations)):
    non_optim_dict = dict(zip(non_optim_keys, non_optim_values))
    config = {**non_optim_dict, **optim_dict}
    param_combinations.append((i, config))  # Store index with config

# Load dataset once
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(np.int64)

val_size = int(0.15 * len(X))

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# Find last completed index
completed_indices = {
    int(f.split("_")[1]) for f in os.listdir("checkpoints") if f.startswith("experiment_") and f.endswith(".pt")
}
last_completed_index = max(completed_indices, default=-1)

# Loop through all parameter configurations
for index, config in param_combinations:
    if index <= last_completed_index:
        print(f"Skipping experiment {index}, already completed.")
        continue  # Skip already completed runs

    config.update({
        'dataset': 'mnist',
        'extra_train': 1e-4,
        'training_step': 'classification_step',
        'scheduler': 'StepLR',
        'scheduler_config': {'step_size': 2000, 'gamma': 1},
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'validation_step': 'classification_step',
    })

    # Generate a unique experiment name
    experiment_name = f"experiment_{index}_mnist_{config['optimizer']}_eps{config['epsilon']}_bs{config['batch_size']}_lr{config['optimizer_config'].get('lr', 'NA')}"

    # Create TensorBoard writer for this experiment
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}", flush_secs=10)

    # Log hyperparameters including index
    flat_hparams = {
        'index': index,
        'epsilon': config['epsilon'],
        'batch_size': config['batch_size'],
        'n_components': config['n_components'],
        'optimizer': config['optimizer'],
    }
    for key, value in config['optimizer_config'].items():
        flat_hparams[f'optimizer_{key}'] = value
    for key, value in config['scheduler_config'].items():
        flat_hparams[f'scheduler_{key}'] = value

    writer.add_hparams(flat_hparams, metric_dict={})

    # Model setup
    model = torch.hub.load(
        'cat-claws/nn', 'simplecnn', 
        convs=[(1, 16, 5), (16, 24, 5)], 
        linears=[24*4*4, 100]
    ).to(config['device'])

    for k, v in config.items():
        if k.endswith('_step'):
            config[k] = eval(v)
    #     elif k == 'optimizer':
    #         config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
    #         config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

    # Resume training if checkpoint exists
    last_checkpoint = max(
        (int(f.split("_")[-1].split(".pt")[0]) for f in os.listdir("checkpoints") if f.startswith(f"experiment_{index}") and f.endswith(".pt")),
        default=-1
    )

    if last_checkpoint >= 0:
        checkpoint_path = f"checkpoints/{experiment_name}_{last_checkpoint:03}.pt"
        print(f"Resuming {experiment_name} from epoch {last_checkpoint}")
        model.load_state_dict(torch.load(checkpoint_path))

    # Initialize optimizer and scheduler
    OptimizerClass = getattr(torch.optim, config['optimizer'])
    config['optimizer'] = OptimizerClass([p for p in model.parameters() if p.requires_grad], **config['optimizer_config'])
    config['scheduler'] = torch.optim.lr_scheduler.StepLR(config['optimizer'], **config['scheduler_config'])
    print(config)

    # Split dataset
    extra_size = int(config['extra_train'] * (len(X) - val_size))
    train_indices, extra_indices, val_indices = torch.utils.data.random_split(
        range(len(X)), [len(X) - val_size - extra_size, extra_size, val_size]
    )

    X_train, y_train = X[train_indices], y[train_indices]
    X_extra, y_extra = X[extra_indices], y[extra_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    # Apply shift transformation
    shift = shift_towards_nearest_other_class(
        X_extra, y_extra, X_extra, y_extra, 
        n_components=config['n_components'], 
        epsilon=config['epsilon']
    )
    X_private = np.clip(X_extra + shift, 0, 1)
    X_train = np.vstack((X_train, X_private)).reshape(-1, 1, 28, 28)
    y_train = np.hstack((y_train, y_extra))
    X_val = X_val.reshape(-1, 1, 28, 28)

    # Create PyTorch datasets
    train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
    val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=4, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, num_workers=4, batch_size=config['batch_size'])

    # Training loop (resuming from last epoch)
    for epoch in range(last_checkpoint + 1, 100):
        train(model, train_loader=train_loader, epoch=epoch, writer=writer, **config)
        validate(model, val_loader=val_loader, epoch=epoch, writer=writer, **config)

        checkpoint_path = f"checkpoints/{experiment_name}_{epoch:03}.pt"
        torch.save(model.state_dict(), checkpoint_path)

    # Log final predictions
    outputs = predict(model, predict_classification_step, val_loader=val_loader, **config)

    writer.flush()
    writer.close()

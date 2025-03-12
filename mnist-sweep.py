import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

from shift import shift_towards_nearest_other_class
import itertools

param_grid = {
    'epsilon': [1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99999],  # Sweep over epsilon
    'batch_size': [16, 32, 64],  # Sweep over batch size
    'learning_rate': [1.0, 0.1, 0.01],  # Sweep over learning rate
    'n_components': [2, 16, 256],  # Sweep over PCA components
}

param_combinations = list(itertools.product(*param_grid.values()))


for params in param_combinations:
    # Unpack parameters
    config = dict(zip(param_grid.keys(), params))
    config.update({
        'dataset': 'mnist',
        'extra_train': 1e-4,  # Keeping it constant
        'training_step': 'classification_step',
        'optimizer': 'Adadelta',
        'scheduler': 'StepLR',
        'scheduler_config': {'step_size': 2000, 'gamma': 1},
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'validation_step': 'classification_step',
    })

	for model in [
		torch.hub.load('cat-claws/nn', 'simplecnn', convs = [], linears = [13, 64, 32, 16, 8, 4], num_classes = 1).to(config['device']),
		torch.hub.load('cat-claws/nn', 'simplecnn', convs = [ (1, 16, 5), (16, 24, 5) ], linears = [24*4*4, 100]).to(config['device']),
		torch.hub.load('cat-claws/nn', 'exampleconvnet', in_channels = 1)
	]:


config = {
	'dataset':'mnist',
    'extra_train':1e-4,
    'epsilon':1e-4,
    'n_components':3,
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
		}


		writer = SummaryWriter(comment = f"_{config['dataset']}_{model._get_name()}_{config['training_step']}_{config['epsilon']}_{config['extra_train']}", flush_secs=10)

		for k, v in config.items():
			if k.endswith('_step'):
				config[k] = eval(v)
			elif k == 'optimizer':
				config[k] = vars(torch.optim)[v]([p for p in model.parameters() if p.requires_grad], **config[k+'_config'])
				config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])		

		mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='liac-arff')
		X = mnist.data.astype(np.float32) / 255.0
		y = mnist.target.astype(np.int64)

		val_size = int(0.15 * len(X))
		extra_size = int(config['extra_train'] * (len(X) - val_size))

		train_indices, extra_indices, val_indices = torch.utils.data.random_split(range(len(X)), [len(X) - val_size - extra_size, extra_size, val_size])

		X_train, y_train = X[train_indices], y[train_indices]
		X_extra, y_extra = X[extra_indices], y[extra_indices]
		X_val, y_val = X[val_indices], y[val_indices]

		shift = shift_towards_nearest_other_class(X_extra, y_extra, X_extra, y_extra, n_components = config['n_components'], epsilon = config['epsilon'])

		X_private = np.clip(X_extra + shift, 0, 1)

		X_train = np.vstack((X_train, X_private)).reshape(-1, 1, 28, 28)
		y_train = np.hstack((y_train, y_extra))

		X_val = X_val.reshape(-1, 1, 28, 28)

		train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
		val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.int64))

		train_loader = torch.utils.data.DataLoader(train_set, num_workers = 4, batch_size = config['batch_size'], shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_set, num_workers = 4, batch_size = config['batch_size'])


		for epoch in range(100):
			if epoch > 0:
				train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

			validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

			torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

		print(model)

		outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

		# print(outputs.keys(), outputs['predictions'])

		writer.flush()
		writer.close()
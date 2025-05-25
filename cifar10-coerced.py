import torch
from torch.utils.tensorboard import SummaryWriter

from torchiteration import train, validate, predict, classification_step, predict_classification_step, build_optimizer, build_scheduler, save_hparams, attacked_classification_step

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--optimizer', type=str)
parser.add_argument('--scheduler', type=str)

parser.add_argument('--model-depth', type=int)
parser.add_argument('--model-widen_factor', type=int)
parser.add_argument('--model-drop_rate', type=float)
parser.add_argument('--model-layers', type=int, nargs='+')
parser.add_argument('--model-block', type=str)

parser.add_argument('--optimizer-lr', type=float)
parser.add_argument('--optimizer-momentum', type=float)
parser.add_argument('--optimizer-weight_decay', type=float)
parser.add_argument('--optimizer-nesterov', action='store_true')
parser.add_argument('--optimizer-betas', type=lambda s: tuple(map(float, s.split(','))))
parser.add_argument('--optimizer-eps', type=float)

parser.add_argument('--scheduler-T_max', type=int)
parser.add_argument('--scheduler-eta_min', type=float)
parser.add_argument('--scheduler-step_size', type=int)
parser.add_argument('--scheduler-gamma', type=float)

parser.add_argument('--dataset', type=str)
parser.add_argument('--path', type=str)

parser.add_argument('--train_transform', type=str)
parser.add_argument('--test_transform', type=str)

parser.add_argument('--private_ratio', type=float)
parser.add_argument('--n_components', type=int)
parser.add_argument('--training_step', type=str)
parser.add_argument('--validation_step', type=str, default='classification_step')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--posion-eps', type=float)
parser.add_argument('--note', type=str)

parser.add_argument('--atk', type=str)
parser.add_argument('--atk-eps', type=float)
parser.add_argument('--atk-alpha', type=float)
parser.add_argument('--atk-steps', type=int)


config = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
print(config)

writer = SummaryWriter(comment = f"_{config['dataset']}_{config['model']}", flush_secs=10)
save_hparams(writer, config, metric_dict={'Epoch-correct/valid': 0})


model = torch.hub.load('cat-claws/nn', config['model'], pretrained= False, **{k[6:]: config.pop(k) for k in list(config) if k.startswith('model_')}).to(config['device'])

config.update({k: eval(v) for k, v in config.items() if k.endswith('_step')})
config['optimizer'] = build_optimizer(config, [p for p in model.parameters() if p.requires_grad])
config['scheduler'] = build_scheduler(config, config['optimizer'])

if 'atk' in config:
    from utils import build_atk
    config['atk'] = build_atk(config, model)

from hardcoded_transforms import transforms

import torchvision
# train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms(config.pop('train_ransform', None)))

datasets = [
    torch.hub.load('cat-claws/datasets', 'CIFAR10', path = 'cat-claws/poison', name = 'cifar10', split='train', transform = transforms(config.get('train_transform', None))),
    torch.hub.load('cat-claws/datasets', 'CIFAR10', path = 'cat-claws/'+config['path'], name = config['dataset'], split='train', transform = transforms(config.pop('train_transform', None))) if config['dataset'] != '' else None
]
from mix import MultiDatasetMixer
train_set = MultiDatasetMixer(datasets, [1 - config['private_ratio'], config['private_ratio']], seed=123)
print('train length: ', len(train_set))

val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms(config.pop('test_transform', None)))

train_loader =  torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
val_loader =  torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

for epoch in range(config['epochs']):

    if epoch > 0:
        train(model, train_loader = train_loader, epoch = epoch, writer = writer, **config)

    validate(model, val_loader = val_loader, epoch = epoch, writer = writer, **config)

    # torch.save(model.state_dict(), 'checkpoints/' + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(model)

outputs = predict(model, predict_classification_step, val_loader = val_loader, **config)

# print(outputs.keys(), outputs['predictions'])

writer.flush()
writer.close()
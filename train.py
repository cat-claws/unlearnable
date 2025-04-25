import argparse
import inspect
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torchvision.models as models
import numpy as np
from torch.utils.tensorboard.summary import hparams

def parse_args():
    parser = argparse.ArgumentParser()

    # Core components
    parser.add_argument('--model', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--scheduler', type=str)

    # Prefixed model args
    parser.add_argument('--model-pretrained', action='store_true')
    parser.add_argument('--model-num_classes', type=int)
    parser.add_argument('--model-in_channels', type=int)

    # Optimizer args
    parser.add_argument('--optimizer-lr', type=float)
    parser.add_argument('--optimizer-momentum', type=float)
    parser.add_argument('--optimizer-weight_decay', type=float)
    parser.add_argument('--optimizer-nesterov', action='store_true')
    parser.add_argument('--optimizer-betas', type=lambda s: tuple(map(float, s.split(','))))
    parser.add_argument('--optimizer-eps', type=float)

    # Scheduler args
    parser.add_argument('--scheduler-T_max', type=int)
    parser.add_argument('--scheduler-eta_min', type=float)
    parser.add_argument('--scheduler-step_size', type=int)
    parser.add_argument('--scheduler-gamma', type=float)

    # Training args
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--extra_train', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--training_step', type=str)
    parser.add_argument('--validation_step', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str)

    return parser.parse_args()

import inspect

def get_kwargs_for(cls, config, prefix):
    sig = inspect.signature(cls)
    return {
        k[len(prefix)+1:]: config.pop(k)
        for k in list(config)
        if k.startswith(prefix + "_") and k[len(prefix)+1:] in sig.parameters
    }
    
import torchvision.models as models

def build_model(config):
    cls = getattr(models, config['model'])
    return cls(**get_kwargs_for(cls, config, "model"))

def build_optimizer(config, params):
    cls = getattr(torch.optim, config['optimizer'])
    return cls(params, **get_kwargs_for(cls, config, "optimizer"))

def build_scheduler(config, optimizer):
    cls = getattr(schetorch.optim.lr_schedulerd, config['scheduler'])
    return cls(optimizer, **get_kwargs_for(cls, config, "scheduler"))


def format_for_hparams(config):
    out = {}
    for k, v in config.items():
        if isinstance(v, (list, tuple, set)):
            v = torch.tensor(list(v)) if all(isinstance(i, (int, float, bool)) for i in v) else np.array(list(v), dtype=object)
        elif isinstance(v, np.ndarray):
            v = torch.tensor(v) if np.issubdtype(v.dtype, np.number) else v
        out[k] = v
    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Core components
    parser.add_argument('--model', type=str)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--scheduler', type=str)

    # Prefixed model args
    parser.add_argument('--model-pretrained', action='store_true')
    parser.add_argument('--model-num_classes', type=int)
    parser.add_argument('--model-in_channels', type=int)

    # Optimizer args
    parser.add_argument('--optimizer-lr', type=float)
    parser.add_argument('--optimizer-momentum', type=float)
    parser.add_argument('--optimizer-weight_decay', type=float)
    parser.add_argument('--optimizer-nesterov', action='store_true')
    parser.add_argument('--optimizer-betas', type=lambda s: tuple(map(float, s.split(','))))
    parser.add_argument('--optimizer-eps', type=float)

    # Scheduler args
    parser.add_argument('--scheduler-T_max', type=int)
    parser.add_argument('--scheduler-eta_min', type=float)
    parser.add_argument('--scheduler-step_size', type=int)
    parser.add_argument('--scheduler-gamma', type=float)

    # Training args
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--extra_train', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--training_step', type=str)
    parser.add_argument('--validation_step', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str)

    config = {k: v for k, v in vars(parse_args()).items() if v is not None}
    print(config)
    # model = instantiate(models, config.get('model'), config, 'model')
    # params = list(model.parameters()) if model else None
    model = build_model(config)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)

    if model: print(f"✅ Model: {model.__class__.__name__}")

    # optimizer = instantiate(optim, config.get('optimizer'), config, 'optimizer', params) if params else None
    if optimizer: print(f"✅ Optimizer: {optimizer}")

    # scheduler = instantiate(sched, config.get('scheduler'), config, 'scheduler', optimizer) if optimizer else None
    if scheduler: print(f"✅ Scheduler: {scheduler}")

    formatted = format_for_hparams(config)
    exp, ssi, sei = hparams(hparam_dict=formatted, metric_dict={'Epoch-correct/valid': 0})
    print("\n✅ hparams summaries ready.")

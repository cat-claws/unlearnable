import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

import steps
import sampling
from utils import nets, datasets, iterate, misc
from ensemble import SE

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--run', type=str, default='Dec18_11-01-23_vm1_CIFAR10_ResNet_our_step')
# parser.add_argument('--test_attacks', type=str, nargs='+', default=['PGD_Linf'])
# parser.add_argument('--hparams_seed', type=int, default=0, help='Seed for hyperparameters')
args = parser.parse_args()

import pandas as pd

df = pd.read_json("checkpoints/configs.json", lines = True)
config = (df.loc[df['run'] == args.run]).to_dict('records')[0]
config.update({
	'num':5,	
# 	'attack':'PGD',
# 	'attack_config':{
# 		'eps':8/255,
# 		'alpha':1/255,
# 		'steps':20,
# 		'random_start':False,
# 	},
"validation_step": "ordinary_step",
})
import json
print(json.dumps(config, indent=4))



_, val_set, channel = misc.auto_sets(config['dataset'])

		
if 'CIFARResNet' not in config['run']:
	m_ = nets.auto_net(channel).cuda()
else:
	import pytorchcv.model_provider
	m_ = pytorchcv.model_provider.get_model(f"resnet20_{config['dataset'].lower()}", pretrained=False).to(config['device'])

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'sample_':
		config[k] = vars(sampling)[v]
		m = SE(m_, **config)
	# elif k == 'optimizer':
	# 	config[k] = vars(torch.optim)[v](m.parameters(), **config[k+'_config'])
	# 	config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])
	elif k == 'adversarial' or k == 'attack':
		config[k] = vars(torchattacks)[v](m, **config[k+'_config'])

m.net.load_state_dict(torch.load("checkpoints_/" + config['run'] + f"_{0:03}.pt"))
writer = SummaryWriter(comment = f"_SE_{config['run']}")
# writer.add_hparams(config, {})





for epoch in range(10, 300, 10):

	m.net.load_state_dict(torch.load("checkpoints_/" + config['run'] + f"_{epoch:03}.pt"))

	iterate.attack(m,
		val_set = val_set,
		epoch = epoch,
		writer = writer,
		atk = config['attack'],
		**config
	)



writer.flush()
writer.close()

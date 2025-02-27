import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

import steps
import sampling
from utils import nets, datasets, iterate, misc

config = {
	'dataset':'SVHN',
	'training_step':'our_step',
	'z':1,
	'batch_size':32,
	'optimizer':'Adadelta',
	'optimizer_config':{
		'lr':1,
		# 'momentum':0.5,
		# 'weight_decay':2e-4,
	},
	'scheduler':'MultiStepLR',
	'scheduler_config':{
		'milestones':[75,90,105, 120, 135, 150],
		'gamma':1
	},
	# 'scheduler':'StepLR',
	# 'scheduler_config':{
	# 	'step_size':15,
	# 	'gamma':0.1,
	# },
	# 'noise_level':0.6,
	'sample_':'sample_uniform_linf_with_clamp',
	'num':50,	
	'eps':8/255,
	'attack':'PGD',
	'attack_config':{
		'eps':8/255,
		'alpha':1/255,
		'steps':20,
		'random_start':False,
	},
	# 'attack':'PGDL2',
	# 'attack_config':{
	# 	'eps':0.5, #PGD
	# 	'alpha':0.2,
	# 	'steps':40,
	# 	'random_start':True,
	# }
	'microbatch_size':10000,
	'threshold':0.95,
	'adversarial':'TPGD',
	'adversarial_config':{
		'eps':8/255,
		'alpha':2/255,
		'steps':10,
	},
	'device':'cuda',
	'validation_step':'augmented_step',
	'attacked_step':'attacked_step'
}

train_set, val_set, channel = misc.auto_sets(config['dataset'])
# m = nets.auto_net(channel).cuda()
# m.load_state_dict(torch.load('checkpoints/Dec14_02-53-38_ruihan-MS-7B23_SVHN_ResNet_trades_step_005.pt'))
import pytorchcv.model_provider
m = pytorchcv.model_provider.get_model(f"resnet20_{config['dataset'].lower()}", pretrained=True).to(config['device'])
# m.apply(misc.weight_init)
m.features[0:2].apply(misc.weight_init)
for name, param in m.named_parameters():                
	if not (name.startswith('features.init_block.') or name.startswith('features.stage1.')):
		param.requires_grad = False

writer = SummaryWriter(comment = f"_{config['dataset']}_{m._get_name()}_{config['training_step']}")
# writer.add_hparams(config, {})

import json
with open("checkpoints/configs.json", 'a') as f:
	f.write(json.dumps({**{'run':writer.log_dir.split('/')[-1]}, **config}) + '\n')
	print(json.dumps(config, indent=4))

for k, v in config.items():
	if k.endswith('_step'):
		config[k] = vars(steps)[v]
	elif k == 'sample_':
		config[k] = vars(sampling)[v]
	elif k == 'optimizer':
		config[k] = vars(torch.optim)[v](m.parameters(), **config[k+'_config'])
		config['scheduler'] = vars(torch.optim.lr_scheduler)[config['scheduler']](config[k], **config['scheduler_config'])
	elif k == 'adversarial' or k == 'attack':
		config[k] = vars(torchattacks)[v](m, **config[k+'_config'])
		

for epoch in range(300):
	if epoch > 0:
		iterate.train(m,
			train_set = val_set,
			epoch = epoch,
			writer = writer,
			atk = config['adversarial'],
			**config
		)

	iterate.attack(m,
		val_set = val_set,
		epoch = epoch,
		writer = writer,
		atk = config['attack'],
		**config
	)

	torch.save(m.state_dict(), "checkpoints_/" + writer.log_dir.split('/')[-1] + f"_{epoch:03}.pt")

print(m)

outputs = iterate.predict(m,
	steps.predict_step,
	val_set = val_set,
	**config
)

# print(outputs.keys(), outputs['predictions'])
writer.flush()
writer.close()

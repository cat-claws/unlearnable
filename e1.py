import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchattacks

import steps
import sampling
from utils import nets, datasets, iterate, misc

config = {
	'dataset':'MNIST',
	'training_step':'our_step',
	'z':1,
	'batch_size':32,
	# 'optimizer':'SGD',
	# 'optimizer_config':{
	# 	'lr':0.1,
	# 	'momentum':0.9,
	# 	'weight_decay':2e-4,
	# },
	'optimizer':'Adadelta',
	'optimizer_config':{
		'lr':1,
	},
	# 'optimizer':'AdamW',
	# 'optimizer_config':{
	# 	'weight_decay':0,
	# },
	'scheduler':'MultiStepLR',
	'scheduler_config':{
		'milestones':[75,90,100],
		'gamma':1,
	},
	# 'noise_level':0.6,
	'sample_':'sample_uniform_linf_with_clamp',
	'num':50,	
	'eps':0.3,
	'attack':'PGD',
	'attack_config':{
		'eps':0.3,
		'alpha':0.1,
		'steps':10,
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
		'eps':0.3,
		'alpha':0.1,
		'steps':7,
	},
	'device':'cuda',
	'validation_step':'augmented_step',
	'attacked_step':'attacked_step'
}

train_set, val_set, channel = misc.auto_sets(config['dataset'])
m = nets.auto_net(channel).cuda()
m.load_state_dict(torch.load('checkpoints/Dec11_16-08-12_vm1_MNIST_ConvNet_our_step_269.pt'))


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
			train_set = train_set,
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

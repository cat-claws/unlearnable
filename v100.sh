# #!/bin/bash

# bash cluster.sh

# srun --partition=researchlong --gres=gpu:1 --constraint=a5000 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet20_svhn --training_step classification_step --optimizer SGD --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 60

# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-coerced.py --model resnet_cifar  --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR --scheduler-T_max 200 --epochs 201 --note "simple cifar resnet"


srun --partition=sunjunresearch --gres=gpu:1 --constraint=h100nvl python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet18_cifar --training_step attacked_classification_step  --atk PGD --atk-eps 0.03137254901 --atk-alpha 0.00784313725 --atk-steps 10 --optimizer AdamW --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model wideresnet --model-depth 34 --model-widen_factor 10 --model-drop_rate 0.3 --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201 --training_step classification_step

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-8-huang2021unlearnable' --model resnet18_cifar --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201



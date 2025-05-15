#!/bin/bash

# srun --partition=researchlong --gres=gpu:1 --constraint=a5000 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet20_svhn --training_step classification_step --optimizer SGD --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 60

# srun --partition=researchlong --gres=gpu:1 --constraint=a100 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet18_cifar --training_step classification_step --optimizer Adam --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201 --note "gaussian noise"


# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-coerced.py --model resnet_cifar  --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR --scheduler-T_max 200 --epochs 201 --note "simple cifar resnet"


# srun --partition=researchlong --gres=gpu:1 --constraint=a100 python cifar10-coerced.py  --dataset 'cifar10-4-wen2023adversarial-pull' --model resnet18_cifar --training_step attacked_classification_step  --atk PGD --atk-eps 0.0156862745 --atk-alpha 0.0031372549 --atk-steps 10 --optimizer AdamW --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201

# srun --partition=researchlong --gres=gpu:1 --constraint=a40 python -u learn-similar.py  

# srun --partition=researchlong --gres=gpu:1 --constraint=a5000 python learn-info.py  

# srun --partition=researchlong --gres=gpu:1 --constraint=v100-32gb python -u learn-simclr.py  

# 0.03137254901 0.00784313725
# 0.06274509802 0.0156862745

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model wideresnet --model-depth 34 --model-widen_factor 10 --model-drop_rate 0.3 --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201 --training_step classification_step

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-8-huang2021unlearnable' --model resnet18_cifar --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201


# srun --partition=researchlong --gres=gpu:1 --constraint=h100 python -u cifar10.py  

srun --partition=sunjunresearch --gres=gpu:1 python -u cifar10-erm.py  


read -p "Press Enter to exit..."
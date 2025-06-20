#!/bin/bash

# srun --partition=researchlong --gres=gpu:1 --constraint=a5000 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet20_svhn --training_step classification_step --optimizer SGD --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 60

# srun --partition=researchlong --gres=gpu:1 --constraint=a100 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model resnet18_cifar --training_step classification_step --optimizer Adam --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201 --note "gaussian noise"


# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-coerced.py --model resnet_cifar  --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR --scheduler-T_max 200 --epochs 201 --note "simple cifar resnet"


# srun --partition=researchlong --gres=gpu:1 --constraint=a100 python cifar10-coerced.py  --dataset 'cifar10-4-wen2023adversarial-pull' --model resnet_cifar --training_step attacked_classification_step  --atk PGD --atk-eps 0.0156862745 --atk-alpha 0.0031372549 --atk-steps 10 --optimizer AdamW --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201

# srun --partition=researchlong --gres=gpu:1 --constraint=a40 python -u learn-similar.py  

# srun --partition=researchlong --gres=gpu:1 --constraint=a5000 python learn-info.py  

# srun --partition=researchlong --gres=gpu:1 --constraint=v100-32gb python -u learn-simclr.py  

# 0.03137254901 0.00784313725
# 0.06274509802 0.0156862745

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-16-huang2021unlearnable' --model wideresnet --model-depth 34 --model-widen_factor 10 --model-drop_rate 0.3 --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201 --training_step classification_step

# srun --partition=researchlong --gres=gpu:1 python cifar10-coerced.py --dataset 'cifar10-8-huang2021unlearnable' --model resnet18_cifar --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR  --scheduler-T_max 200 --epochs 201


# srun --partition=researchlong --gres=gpu:1 --constraint=h100 python -u cifar10.py  

# srun --partition=sunjunresearch --gres=gpu:1 python -u cifar10-erm.py  

srun --partition=sunjunresearch --gres=gpu:2 python -u beat.py  

# srun --partition=researchlong --gres=gpu:1 --constraint=a100 python -u cifar10-opposite-3.py

# srun --partition=sunjunresearch --gres=gpu:1  python -u cifar10-opposite-3.py

# srun --partition=researchlong --gres=gpu:1 --constraint=l40 python -u cifar10-coerced.py --dataset 'cifar10-8-8-wen2023adversarial-pull' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 30 --epochs 31 --private_ratio 0.1 --train_transform cifar10_T --test_transform cifar10_T


#srun --partition=researchlong --gres=gpu:1 --constraint=l40 python -u cifar10-coerced.py --dataset 'resnet18-retrain' --path trial --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 30 --epochs 31 --private_ratio 0.1 --train_transform cifar10_T --test_transform cifar10_T

# srun --partition=researchlong --gres=gpu:1 --constraint=l40 python -u tinyimagenet-coerced.py --dataset '' --path trial --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --model-num_classes 200 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 30 --epochs 31 --private_ratio 0.1 --train_transform cifar10_T --test_transform cifar10_T

# srun --partition=sunjunresearch --gres=gpu:1 python cifar-10-retrain.py >> notes2.txt

read -p "Press Enter to exit..."

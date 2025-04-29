#!/bin/bash

./cluster.sh

# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-example.py --model resnet18_cifar  --optimizer AdamW --scheduler CosineAnnealingLR --scheduler-T_max 200 --epsilon 0.000001 --extra_train 0.01 --n_components 8

# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-example.py --model wideresnet --model-depth 28 --model-widen_factor 10 --model-drop_rate 0.3 --optimizer AdamW --scheduler CosineAnnealingLR --scheduler-T_max 200 --epsilon 0.000001 --extra_train 0.01 --n_components 100

# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-example.py --model wideresnet --model-depth 40 --model-widen_factor 10 --model-drop_rate 0.5 --optimizer SGD --optimizer-lr 0.1 --optimizer-momentum 0.9 --optimizer-weight_decay 0.0005 --scheduler CosineAnnealingLR --scheduler-T_max 400 --epsilon 0.000001 --extra_train 0.01 --n_components 16 --batch_size 512 --optimizer-nesterov --epochs 401

# srun --partition=sunjunresearch --gres=gpu:1 python wide.py

# srun --partition=sunjunresearch --gres=gpu:1 python cifar10-example3.py

srun --partition=sunjunresearch --gres=gpu:1 python cifar10-coerced.py --model resnet_cifar  --optimizer SGD --optimizer-lr 0.1 --scheduler CosineAnnealingLR --scheduler-T_max 200 --epochs 201 --note "simple cifar resnet"

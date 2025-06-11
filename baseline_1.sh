#cifar10-8-8-wen2023adversarial-pull
#cifar10-8-huang2021unlearnable
#resnet18-ad2-3-8sgd-smallbatch

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.001 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.01 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.05 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.1 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.2 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.3 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.4 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.5 --train_transform cifar10_T --test_transform cifar10_T

CUDA_VISIBLE_DEVICES=0 python -u cifar10-ft.py --dataset 'cifar10-8-huang2021unlearnable' --path poison --model resnet_cifar --model-block "" --model-layers 2 2 2 2 --training_step classification_step --optimizer SGD --optimizer-lr 0.1 --optimizer-weight_decay 0.0005 --optimizer-momentum 0.9 --scheduler CosineAnnealingLR  --scheduler-T_max 60 --epochs 56 --epochs-ft 5 --private_ratio 0.9 --train_transform cifar10_T --test_transform cifar10_T


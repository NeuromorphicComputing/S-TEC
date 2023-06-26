# Self-supervised Learning Through Efference Copies

This repository is the official implementation of [Self-supervised Learning Through Efference Copies (S-TEC)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1d1cea122b9ec9f78acc21510659e500-Abstract-Conference.html), accepted at NeurIPS 2022.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

### CIFAR-10
To train a ResNet-18 with S-TEC on CIFAR-10, run this command:

```train
python train.py --dataset cifar10 --data_dir <path-to-data> --arch resnet18 --temperature .5 \
--learning_rate 4. --manip_lambda 1. --feat_dim 64 --results_path ./s_tec_cifar10_resnet18
```

To train a ResNet-50 with S-TEC on CIFAR-10, run this command:

```train
python train.py --dataset cifar10 --data_dir <path-to-data> --arch resnet50 --temperature .5 \
--learning_rate 4. --manip_lambda 1. --hidden_mlp 1024 --feat_dim 64 \
--results_path ./s_tec_cifar10_resnet50
```

### CIFAR-100
To train instead on CIFAR-100, replace `cifar10` with `cifar100`. 

### STL-10
For training on STL-10, likely multiple GPUs are necessary to achieve a total batch size of 1024. For training a ResNet-18, use the command below (assuming 2 GPUs):
```train
python train.py --dataset stl10 --data_dir <path-to-data> --arch resnet18 --temperature .2 \
--learning_rate 1.2 --manip_lambda .3 --results_path ./s_tec_stl10_resnet18 \
--use_solarization 1 --use_gaussian_blur 1 --batch_size 512 --gpus 2
```

Subsequently, training the linear classifier is performed separately using this command:
```
python train.py --dataset stl10 --data_dir <path-to-data> --arch resnet18 --batch_size 1024 \
 --max_epochs 100 --optimizer sgd --learning_rate .04 --stop_gradient 2 \
 --reinitialize_supervised_head 1 --nesterov 1 --p_color_jitter -1 --p_grayscale -1 \
 --ckpt_path <path-to-checkpoint> --results_path ./s_tec_stl10_resnet18_linear_fit
```

To train a ResNet-50, replace all above occurrences of `resnet18` with `resnet50` and adapt the number of GPUs and batch size per GPU accordingly.

### ImageNet
Training a ResNet-50 on ImageNet can be achieved by executing following command (e.g. using 8 GPUs):

```train
python train.py --dataset imagenet--data_dir <path-to-data> --arch resnet50 --temperature .1 \
--learning_rate 1.97 --manip_lambda .6 --results_path ./s_tec_imagenet_resnet50 \
--use_solarization 1 --use_gaussian_blur 1 --batch_size 210 --gpus 8 --max_epochs 100
```

## Evaluation

Evaluation is performed using the following command:

```
python eval.py --dataset <dataset> --data_dir <path-to-data> --arch <resnetXY> --ckpt_path <path-to-checkpoint>
```

See commands in section pre-trained models for examples.


## Pre-trained Models

We include here pretrained models (trained using S-TEC and SimCLR) for ResNet-18 on CIFAR-100, and plan to release all other models in a public version.

To evaluate a model trained with S-TEC, execute:

```
python eval.py --dataset cifar100 --data_dir <path-to-data> --arch resnet18 \
--ckpt_path pretrained/resnet18_stec_cifar100.ckpt
```

It should print `0.6680` (i.e. `66.80%`).

To evaluate a model trained with SimCLR, execute:

```
python eval.py --dataset cifar100 --data_dir <path-to-data> --arch resnet18 \
--ckpt_path pretrained/resnet18_simclr_cifar100.ckpt
```

It should print `0.6540` (i.e. `65.40%`).

import os
import json
import argparse

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

import utils
from train import STec
from cifar100_datamodule import CIFAR100DataModule


def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--discrimination_lambda', type=float, default=2.)
    parser.add_argument('--manip_lambda', type=float, default=.0)
    parser.add_argument('--supervised_lambda', type=float, default=1.)
    parser.add_argument('--stop_gradient', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--n_bins', type=int, default=6)
    parser.add_argument('--lr_scheduler', type=str, default='none', choices=['none', 'cosine'])
    parser.add_argument('--nesterov', type=int, default=0)
    parser.add_argument('--manip_hidden_mlp', type=str, default='512')

    parser.add_argument('--reinitialize_supervised_head', type=int, default=0)

    parser.add_argument('--use_solarization', type=int, default=0)
    parser.add_argument('--use_gaussian_blur', type=int, default=0)

    # model args
    parser = STec.add_model_specific_args(parser)
    utils.remove_option(parser, '--batch_size')
    parser.add_argument('--batch_size', type=int, default=1024)
    utils.remove_option(parser, '--hidden_mlp')
    parser.add_argument('--hidden_mlp', type=int, default=-1)

    args = parser.parse_args()
    if args.hidden_mlp < 0:
        if args.arch == 'resnet18':
            args.hidden_mlp = 512
        elif args.arch == 'resnet50':
            args.hidden_mlp = 2048

    val_check_interval = .99
    max_val_steps = 2
    if args.dataset == "stl10":
        dm_test = STL10DataModule(data_dir=args.data_dir, unlabeled_val_split=0, train_val_split=0, num_workers=args.num_workers,
                                  batch_size=args.batch_size, drop_last=False)

        args.num_samples = dm_test.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm_test.size()[-1]

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 0

        dm_test = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=False
        )
        args.num_samples = dm_test.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm_test.size()[-1]

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "cifar100":
        val_split = 0

        dm_test = CIFAR100DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=False
        )
        args.num_samples = dm_test.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm_test.size()[-1]

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        max_val_steps = 1.
        args.maxpool1 = True
        args.first_conv = True

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.online_ft = False

        dm_test = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

        args.num_samples = dm_test.num_samples
        args.input_height = dm_test.size()[-1]
        val_check_interval = .33
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    if args.dataset == 'imagenet':
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        dm_test.test_transforms = test_transform
    else:
        test_transform = torchvision.transforms.ToTensor()
        dm_test.test_transforms = test_transform

    args.num_classes = dm_test.num_classes

    results_path = args.results_path
    if results_path != '':
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, 'eval_flags.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    ckpt_path = None if args.ckpt_path == '' else args.ckpt_path
    if ckpt_path is not None:
        model = STec.load_from_checkpoint(ckpt_path, **args.__dict__, strict=False)
        if args.reinitialize_supervised_head == 1:
            torch.nn.init.normal_(model.supervised_head.linear_layer.weight, std=.01)
            torch.nn.init.zeros_(model.supervised_head.linear_layer.bias)
    else:
        model = STec(**args.__dict__)

    logger = pl.loggers.TensorBoardLogger(results_path)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        fast_dev_run=False,
        logger=logger,
        val_check_interval=val_check_interval,
        limit_val_batches=max_val_steps
    )
    test_results = trainer.test(model, datamodule=dm_test)
    if results_path != '':
        with open(os.path.join(results_path, 'eval_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)


if __name__ == '__main__':
    cli_main()

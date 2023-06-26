import os
import json
import argparse
import contextlib

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR, SyncFunction
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.optimizers.lars import LARS
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

import utils
from hybrid_optimizer_utils import HybridOptim
from transforms import RandomSolarize, RandomGaussian, TransformsSimCLRReturnTransforms
from resnets import resnet18, resnet50
from cifar100_datamodule import CIFAR100DataModule


class BinnedPredictionModel(torch.nn.Module):
    def __init__(self, n_input, n_hidden=(512,), output_shape=(2, 3), n_bins=6):
        super().__init__()
        prev_hidden = n_input
        self.model = torch.nn.Sequential()
        for i, n_h in enumerate(n_hidden):
            self.model.add_module(f'linear_{i}', torch.nn.Linear(prev_hidden, n_h))
            self.model.add_module(f'bn_{i}', torch.nn.BatchNorm1d(n_h))
            self.model.add_module(f'relu_{i}', torch.nn.ReLU())
            prev_hidden = n_h
        self.model.add_module('linear_final', torch.nn.Linear(prev_hidden, n_bins * int(np.prod(output_shape))))
        self.output_shape = output_shape
        self.n_bins = n_bins
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def n_log_prob_and_accuracy(self, features, values, min_values, max_values):
        target_indices = torch.clamp(torch.round(
            (values - min_values) / (max_values - min_values) * self.n_bins - .5).to(
            torch.int64), min=0, max=self.n_bins - 1)
        logits = self.model(features).reshape((-1, self.n_bins, *self.output_shape))
        predicted_indices = torch.argmax(logits, dim=1)
        accuracy = torch.mean(torch.eq(predicted_indices, target_indices).type_as(features))
        losses = torch.nn.functional.cross_entropy(logits, target_indices, reduction='none')
        return losses, accuracy


class Projection(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, final_batch_norm=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = torch.nn.Sequential()
        self.model.add_module('hidden', torch.nn.Linear(self.input_dim, self.hidden_dim))
        self.model.add_module('bn_hidden', torch.nn.BatchNorm1d(self.hidden_dim))
        self.model.add_module('relu_hidden', torch.nn.ReLU())
        self.model.add_module('output', torch.nn.Linear(self.hidden_dim, self.output_dim, bias=False))
        if final_batch_norm:  # As in SimCLR implementation without learnable bias
            _bn = torch.nn.BatchNorm1d(self.output_dim)
            _bn.bias.requires_grad = False
            _bn.bias.zero_()
            self.model.add_module('bn_output', _bn)

    def forward(self, x):
        x = self.model(x)
        return torch.nn.functional.normalize(x, dim=1)


class LinearReadoutModel(torch.nn.Module):
    def __init__(self, n_features, num_classes):
        super().__init__()
        self.linear_layer = torch.nn.Linear(n_features, num_classes, bias=True)
        torch.nn.init.normal_(self.linear_layer.weight, std=.01)
        torch.nn.init.zeros_(self.linear_layer.bias)

    def forward(self, x):
        return self.linear_layer(x)


class STec(SimCLR):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            dataset: str,
            num_nodes: int = 1,
            arch: str = "resnet50",
            hidden_mlp: int = 2048,
            feat_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            temperature: float = 0.1,
            first_conv: bool = True,
            maxpool1: bool = True,
            optimizer: str = "adam",
            exclude_bn_bias: bool = False,
            start_lr: float = 0.0,
            learning_rate: float = 1e-3,
            final_lr: float = 0.0,
            weight_decay: float = 1e-6,
            discrimination_lambda: float = 1.,
            manip_lambda: float = .1,
            **kwargs
    ):
        super().__init__(
            gpus, num_samples, batch_size, dataset, num_nodes, arch, hidden_mlp, feat_dim, warmup_epochs, max_epochs,
            temperature, first_conv, maxpool1, optimizer, exclude_bn_bias, start_lr, learning_rate, final_lr,
            weight_decay, **kwargs
        )

        input_dim = -1
        if self.arch == 'resnet18':
            input_dim = 512
        elif self.arch == 'resnet50':
            input_dim = 2048
        self.projection = Projection(
            input_dim=input_dim,
            hidden_dim=self.hidden_mlp,
            output_dim=self.feat_dim,
            final_batch_norm=bool(kwargs.get('projection_final_batch_norm', 1))
        )

        prediction_input_dim = input_dim * 2

        def _parse_layer_structure(_k):
            n_hidden = (hidden_mlp,)
            if _k in kwargs.keys() and len(kwargs[_k]) > 0:
                n_hidden = list(map(int, kwargs[_k].split(',')))
                if len(n_hidden) <= 1 and n_hidden[0] <= 0:
                    n_hidden = []
            return n_hidden

        manip_model_hidden = _parse_layer_structure('manip_hidden_mlp')

        self.binned_prediction_model = BinnedPredictionModel(
            prediction_input_dim, manip_model_hidden, output_shape=(2, 3), n_bins=kwargs.get('n_bins', 6),
        )
        self.discrimination_lambda = discrimination_lambda
        self.manip_lambda = manip_lambda
        self.supervised_lambda = kwargs.get('supervised_lambda', -1.)
        self.stop_gradient = kwargs.get('stop_gradient', 1)
        self.num_classes = kwargs.get('num_classes', 10)
        self.supervised_head = LinearReadoutModel(input_dim, self.num_classes)

        self.nesterov = bool(kwargs.get('nesterov', 1))
        self.lr_scheduler_name = kwargs.get('lr_scheduler', 'none')
        self.use_gaussian_blur = bool(kwargs.get('use_gaussian_blur', 0))
        self.use_solarization = bool(kwargs.get('use_solarization', 0))
        if self.use_gaussian_blur:
            size = kwargs['input_height']
            min_sigma = .1 * size / 224
            max_sigma = 2. * size / 224
            self.rand_gauss = RandomGaussian(size, p=.5, sigma_range=(min_sigma, max_sigma))
        if self.use_solarization:
            self.solarize = RandomSolarize(p=.2)

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50
        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def shared_step(self, batch):
        supervised_batch = None
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            supervised_batch = batch[1]
            sup_y = supervised_batch[1]
            batch = unlabeled_batch

        x, y = batch
        if self.dataset == 'stl10':
            (sup_img, _), _ = supervised_batch[0]
        img_i, from_i_to_j = x[0]
        img_j, from_j_to_i = x[1]

        gradient_context = torch.no_grad() if self.stop_gradient == 2 else contextlib.nullcontext()
        if self.stop_gradient == 2:
            self.encoder.eval()
        batch_size = img_i.size(0)

        with gradient_context:
            # get h representations, bolts resnet returns a list
            batched_img = torch.cat((img_i, img_j), dim=0)
            if self.use_gaussian_blur:
                batched_img = self.rand_gauss(batched_img)
            if self.use_solarization:
                batched_img, solarize_vector = self.solarize(batched_img, return_selection=True)
            h = self(batched_img)  # prevent leakage through batch norm

            # get z representations
            z = self.projection(h)

            h_i, h_j = h[:batch_size], h[batch_size:]
            z_i, z_j = z[:batch_size], z[batch_size:]

            discrimination_loss, discrimination_accuracy, xent_aux = self.nt_xent_loss_with_accuracy(
                z_i, z_j)

            # S-Tec loss
            prediction_input_i = torch.cat((h_i, h_j), -1)

            # distinguish between cross prediction
            matrix_to_be_predicted = from_i_to_j

            min_values = torch.zeros((1, 2, 3)).type_as(prediction_input_i) - 2.
            max_values = torch.zeros((1, 2, 3)).type_as(prediction_input_i) + 2.
            min_values[..., -1] = - .5
            max_values[..., -1] = + .5
            non_reduced_manip_loss, manip_accuracy = self.binned_prediction_model.n_log_prob_and_accuracy(
                prediction_input_i, matrix_to_be_predicted, min_values, max_values)
            if self.use_solarization:
                s_a, s_b = solarize_vector[:batch_size], solarize_vector[batch_size:]
                cond = torch.logical_not(torch.logical_or(s_a, s_b))
                non_reduced_manip_loss = non_reduced_manip_loss.mean(dim=(1, 2))
                non_reduced_manip_loss = torch.where(
                    cond,
                    non_reduced_manip_loss,
                    torch.zeros_like(non_reduced_manip_loss))
                manip_loss = torch.sum(non_reduced_manip_loss) / torch.sum(cond.type_as(non_reduced_manip_loss))
            else:
                manip_loss = torch.mean(non_reduced_manip_loss)

            loss = discrimination_loss * self.discrimination_lambda
            loss += manip_loss * self.manip_lambda

            aux = dict(
                discrimination=(discrimination_loss, discrimination_accuracy),
                manipulation=(manip_loss, manip_accuracy),
                contrast_accuracy=xent_aux['contrast_accuracy'],
                contrast_entropy=xent_aux['contrast_entropy']
            )

            if self.dataset == 'stl10':
                h_for_supervised_head = self(sup_img)
                y_for_supervised = sup_y
            else:
                h_for_supervised_head = h
                y_for_supervised = torch.cat([y, y], dim=0)

        if self.stop_gradient == 1:
            h_for_supervised_head = h_for_supervised_head.detach()

        logits = self.supervised_head(h_for_supervised_head)
        supervised_loss = torch.nn.functional.cross_entropy(logits, y_for_supervised)
        max_indices = torch.argmax(logits, dim=-1)
        supervised_accuracy = torch.mean(torch.eq(max_indices, y_for_supervised).to(torch.float32))
        aux['supervised'] = (supervised_loss, supervised_accuracy)
        weighted_supervised_loss = self.supervised_lambda * supervised_loss

        if self.supervised_lambda > 0:
            return loss + weighted_supervised_loss, aux
        return loss, aux

    def nt_xent_loss_with_accuracy(self, out_1, out_2, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        cov = cov.to(torch.float32)
        full_batch_size = out_1_dist.size(0)
        batch_size = out_1.size(0)

        world_size = 1
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        dist_indices_1 = torch.arange(batch_size).to(cov.device) + rank * batch_size
        dist_indices_2 = dist_indices_1 + world_size * batch_size

        true_indices = torch.cat([dist_indices_2, dist_indices_1], dim=0)
        self_indices = torch.cat([dist_indices_1, dist_indices_2], dim=0)
        eye_matrix = torch.nn.functional.one_hot(
            self_indices, num_classes=2 * full_batch_size).type_as(cov)
        positive_selection = torch.nn.functional.one_hot(
            true_indices, num_classes=2 * full_batch_size).type_as(cov)

        negative_selection = (1 - positive_selection) * (1 - eye_matrix)

        logit_mask = 1. - (1. - positive_selection) * (1. - negative_selection)
        logit_matrix = cov / self.temperature * logit_mask
        if logit_mask.dtype == torch.float16:
            logit_matrix = logit_matrix - (1. - logit_mask) * 1e3
        else:
            logit_matrix = logit_matrix - (1. - logit_mask) * 1e9

        probabilities = torch.softmax(logit_matrix, dim=1)

        loss_vector = torch.nn.functional.cross_entropy(logit_matrix, true_indices)
        loss = loss_vector.mean()

        predicted_indices = torch.argmax(probabilities, dim=1)
        accuracy = torch.mean(torch.eq(predicted_indices, true_indices).type_as(out_1))

        logits_ab = logit_matrix[:batch_size, full_batch_size:]
        probabilities_ab = torch.softmax(logits_ab, dim=1)
        predicted_indices_ab = torch.argmax(probabilities_ab, dim=1)
        true_indices_ab = true_indices[:batch_size] - full_batch_size
        ab_accuracy = torch.mean(torch.eq(predicted_indices_ab, true_indices_ab).type_as(out_1))
        ab_entropy = torch.mean(torch.sum(-probabilities_ab * torch.log(probabilities_ab + eps), dim=-1))

        aux = dict(
            probabilities=probabilities, logit_matrix=logit_matrix, cov=cov,
            predicted_indices=predicted_indices,
            contrast_accuracy=ab_accuracy, contrast_entropy=ab_entropy
        )

        return loss, accuracy, aux

    def training_step(self, batch, batch_idx):
        loss, aux = self.shared_step(batch)
        discrimination_loss, discrimination_accuracy = aux['discrimination']
        manip_loss, manip_accuracy = aux['manipulation']
        supervised_loss, supervised_accuracy = aux['supervised']

        contrast_accuracy = aux['contrast_accuracy']
        contrast_entropy = aux['contrast_entropy']

        show_accuracy = self.supervised_lambda > 0

        self.log("train/loss", loss, on_step=True, on_epoch=False)
        self.log("train/manip_loss", manip_loss, on_step=True, on_epoch=False)
        self.log("train/manip_accuracy", manip_accuracy, on_step=True, on_epoch=False)
        self.log("train/discrimination_loss", discrimination_loss, on_step=True, on_epoch=False)
        self.log("train/supervised_loss", supervised_loss, on_step=True, on_epoch=False)
        self.log("train/supervised_accuracy", supervised_accuracy, on_step=True, on_epoch=False, prog_bar=show_accuracy)

        self.log("train/contrast_accuracy", contrast_accuracy, on_step=True, on_epoch=False)
        self.log("train/contrast_entropy", contrast_entropy, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, aux = self.shared_step(batch)
        discrimination_loss, discrimination_accuracy = aux['discrimination']
        manip_loss, manip_accuracy = aux['manipulation']
        supervised_loss, supervised_accuracy = aux['supervised']

        contrast_accuracy = aux['contrast_accuracy']
        contrast_entropy = aux['contrast_entropy']

        show_accuracy = self.supervised_lambda > 0

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/manip_loss", manip_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/manip_accuracy", manip_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/discrimination_loss", discrimination_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/supervised_loss", supervised_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/supervised_accuracy", supervised_accuracy, on_step=False, on_epoch=True, sync_dist=True,
                 prog_bar=show_accuracy)

        self.log("val/contrast_accuracy", contrast_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/contrast_entropy", contrast_entropy, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, _batch_idx):
        x, y = batch
        h = self(x)
        logits = self.supervised_head(h)
        supervised_loss = torch.nn.functional.cross_entropy(logits, y)
        max_indices = torch.argmax(logits, dim=-1)
        supervised_accuracy = torch.mean(torch.eq(max_indices, y).to(torch.float32))

        self.log("test/loss", supervised_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/accuracy', supervised_accuracy, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return supervised_loss

    def configure_optimizers(self):
        if self.optim == 'lars':
            params = []
            params_with_weight_decay = []
            excluded_params = []
            skip_list = ['bias', 'bn']
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                elif any(layer_name in name for layer_name in skip_list):
                    excluded_params.append(param)
                elif 'supervised_head' in name:
                    params_with_weight_decay.append(param)
                else:
                    params.append(param)

            optimizer_lars = LARS(  # LARS with weight decay for all parameters
                params,
                lr=self.learning_rate,
                momentum=.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
                nesterov=self.nesterov
            )
            optimizer_sgd = torch.optim.SGD(  # ... except for biases and batch norm parameters, for which SGD is used
                                              #     without weight decay
                excluded_params, lr=self.learning_rate, momentum=.9,
                nesterov=self.nesterov, weight_decay=0.)
            optimizer_sgd_with_weight_decay = torch.optim.SGD(
                params_with_weight_decay, lr=self.learning_rate, momentum=.9,
                nesterov=self.nesterov, weight_decay=self.weight_decay)
            optimizer_list = [optimizer_lars, optimizer_sgd, optimizer_sgd_with_weight_decay]
            hybrid_optimizer = HybridOptim(optimizer_list)
            if self.lr_scheduler_name == 'cosine':
                warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
                total_steps = self.train_iters_per_epoch * self.max_epochs
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(
                        hybrid_optimizer,
                        linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
                return [hybrid_optimizer], [scheduler]
            else:
                return hybrid_optimizer

            # --------------------------------------------------------------------------------

        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=.9, nesterov=self.nesterov)
        else:
            raise ValueError('Unknown optimizer')

        if self.lr_scheduler_name == 'cosine':
            warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
            total_steps = self.train_iters_per_epoch * self.max_epochs
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        elif self.lr_scheduler_name == 'none':
            return optimizer
        else:
            raise ValueError('No such learning rate scheduler')


def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_path', type=str, default='./default_model')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--discrimination_lambda', type=float, default=2.)
    parser.add_argument('--manip_lambda', type=float, default=.0)
    parser.add_argument('--supervised_lambda', type=float, default=1.)
    parser.add_argument('--stop_gradient', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--n_bins', type=int, default=6)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['none', 'cosine'])
    parser.add_argument('--nesterov', type=int, default=0)
    parser.add_argument('--p_grayscale', type=float, default=.2)
    parser.add_argument('--p_color_jitter', type=float, default=.8)
    parser.add_argument('--manip_hidden_mlp', type=str, default='512')
    parser.add_argument('--projection_final_batch_norm', type=int, default=1)

    parser.add_argument('--reinitialize_supervised_head', type=int, default=0)

    parser.add_argument('--use_solarization', type=int, default=0)
    parser.add_argument('--use_gaussian_blur', type=int, default=0)

    # model args
    parser = STec.add_model_specific_args(parser)
    utils.remove_option(parser, '--batch_size')
    parser.add_argument('--batch_size', type=int, default=1024)
    utils.remove_option(parser, '--max_epochs')
    parser.add_argument('--max_epochs', type=int, default=1000)
    utils.remove_option(parser, '--optimizer')
    parser.add_argument('--optimizer', type=str, default='lars', choices=['lars', 'sgd', 'adam'])
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
        dm = STL10DataModule(data_dir=args.data_dir, unlabeled_val_split=0, train_val_split=0, num_workers=args.num_workers,
                             batch_size=args.batch_size, drop_last=True)
        dm_test = STL10DataModule(data_dir=args.data_dir, unlabeled_val_split=0, train_val_split=0, num_workers=args.num_workers,
                                  batch_size=args.batch_size, drop_last=False)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.train_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.size()[-1]

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10":
        val_split = 0

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=True
        )
        dm_test = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=False
        )
        dm.val_dataloader = dm.train_dataloader

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "cifar100":
        val_split = 0

        dm = CIFAR100DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=True
        )
        dm_test = CIFAR100DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split,
            drop_last=False
        )
        dm.val_dataloader = dm.train_dataloader

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.size()[-1]

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet":
        max_val_steps = 1.
        args.maxpool1 = True
        args.first_conv = True

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        args.online_ft = False

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
        dm_test = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
        val_check_interval = .33
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = TransformsSimCLRReturnTransforms(
        args.input_height, jitter_strength=args.jitter_strength,
        p_grayscale=args.p_grayscale, p_color_jitter=args.p_color_jitter)
    dm.val_transforms = TransformsSimCLRReturnTransforms(
        args.input_height, jitter_strength=args.jitter_strength,
        p_grayscale=args.p_grayscale, p_color_jitter=args.p_color_jitter)

    if args.dataset == 'imagenet':
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        dm.test_transforms = test_transform
        dm_test.test_transforms = test_transform
    else:
        test_transform = torchvision.transforms.ToTensor()
        dm.test_transforms = test_transform
        dm_test.test_transforms = test_transform

    args.num_classes = dm.num_classes

    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)
    ckpt_path = None if args.ckpt_path == '' else args.ckpt_path
    if ckpt_path is not None:
        model = STec.load_from_checkpoint(ckpt_path, **args.__dict__, strict=False)
        if args.reinitialize_supervised_head == 1:
            torch.nn.init.normal_(model.supervised_head.linear_layer.weight, std=.01)
            torch.nn.init.zeros_(model.supervised_head.linear_layer.bias)
    else:
        model = STec(**args.__dict__)

    with open(os.path.join(results_path, 'flags.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val/loss", every_n_epochs=10)
    logger = pl.loggers.TensorBoardLogger(results_path)
    callbacks = [lr_monitor, model_checkpoint]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=False,
        logger=logger,
        val_check_interval=val_check_interval,
        limit_val_batches=max_val_steps
    )

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm_test)


if __name__ == '__main__':
    cli_main()

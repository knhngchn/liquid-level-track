import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))
print ("START")

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
parser.add_argument('--fixed', action='store_true')
parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
parser.add_argument('--state_dict', help='load state_dict for continuing training', default=None)
parser.add_argument('--dump_prefix', help='model dump path prefix')
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
parser.add_argument('--lr_pow', help='Learning rate', type=float, default=0.9)
parser.add_argument('--warmup_lr', help='Learning rate', type=float, default=1e-6)
parser.add_argument('--warmup_epochs', help='Learning rate', type=int, default=5)
parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=20)
parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
parser.add_argument('--weight_decay', help='L2 regularization', type=float, default=1e-6)
parser.add_argument('--jit_translate', help='Augment translation', action='store_true')
parser.add_argument('--jit_bright', help='Augment brightness', type=float, default=0)
parser.add_argument('--jit_saturation', help='Augment saturation', type=float, default=0)

parser = parser.parse_args()


# Create the data loaders
if parser.dataset == 'csv':

    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train when training on COCO,')
    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training on COCO,')
    dataset_train = CSVDataset(
        train_file=parser.csv_train,
        class_list=parser.csv_classes,
        transform=transforms.Compose([
            Normalizer(),
            Augmenter(jit_bright=parser.jit_bright, jit_saturation=parser.jit_saturation),
            Resizer(jit_translate=parser.jit_translate),
        ])
    )

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(
            train_file=parser.csv_val,
            class_list=parser.csv_classes,
            transform=transforms.Compose([Normalizer(), Resizer()])
        )

else:
    raise ValueError('Dataset type not understood (must be csv currently), exiting.')

parser.max_iters = int(parser.epochs * len(dataset_train) / parser.batch_size)
parser.warmup_iters = int(parser.warmup_epochs * len(dataset_train) / parser.batch_size)
parser.running_lr = parser.warmup_lr
print('warmup_iters', parser.warmup_iters)
print('max_iters', parser.max_iters)

sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)
if dataset_val is not None:
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=4, collate_fn=collater, batch_sampler=sampler_val)

# Create the model
if parser.depth not in [18, 34, 50, 101, 152]:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
Model = getattr(model, 'ResNet%d' % parser.depth)
retinanet = Model(num_classes=dataset_train.num_classes(), pretrained=True)


if parser.state_dict:
    retinanet.load_state_dict(torch.load(parser.state_dict), strict=False)

retinanet = retinanet.cuda()
retinanet.training = True


# Create optimizer
def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def adjust_learning_rate(optimizer, cur_iter):
    if cur_iter < parser.warmup_iters:
        frac = float(cur_iter) / parser.warmup_iters
        step = parser.lr - parser.warmup_lr
        running_lr = parser.warmup_lr + step * frac
    else:
        frac = (float(cur_iter) - parser.warmup_iters) / (parser.max_iters - parser.warmup_iters)
        scale_running_lr = (1. - min(frac, 0.9999)) ** parser.lr_pow
        running_lr = parser.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr

    return running_lr


if parser.fixed:
    optimizer = optim.Adam(
        [
            *group_weight(retinanet.regressionModel),
            *group_weight(retinanet.classificationModel),
        ],
        lr=parser.lr,
        weight_decay=parser.weight_decay)
else:
    optimizer = optim.Adam(
        group_weight(retinanet),
        lr=parser.lr,
        weight_decay=parser.weight_decay)

total_loss = losses.loss
loss_hist = {'clf': 0, 'reg': 0}
cur_iter = 0

print('Num training images: {}'.format(len(dataset_train)))

for epoch_num in range(parser.epochs):
    print (epoch_num)
    retinanet.train()
    retinanet.freeze_bn()

    for iter_num, data in enumerate(dataloader_train):
        running_lr = adjust_learning_rate(optimizer, cur_iter)
        cur_iter += 1
        try:
            optimizer.zero_grad()
            classification, regression, anchors = retinanet(data['img'].cuda().float())
            classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])
            loss = classification_loss + regression_loss

            if loss == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 3.0, 'inf')
            optimizer.step()

            if cur_iter > 500:
                loss_hist['clf'] += (classification_loss.item() - loss_hist['clf']) / 500
                loss_hist['reg'] += (regression_loss.item() - loss_hist['reg']) / 500
            else:
                loss_hist['clf'] += (classification_loss.item() - loss_hist['clf']) / cur_iter
                loss_hist['reg'] += (regression_loss.item() - loss_hist['reg']) / cur_iter
            loss_hist['total'] = loss_hist['clf'] + loss_hist['reg']
            print('Ep. {} It. {} | lr: {:1.7f} | Clf loss: {:1.5f} | Reg loss: {:1.5f} | Total loss: {:1.5f}'.format(
                epoch_num, iter_num, running_lr, loss_hist['clf'], loss_hist['reg'], loss_hist['total']), flush=True)

        except Exception as e:
            print(e)

    if parser.dataset == 'csv' and parser.csv_val is not None:
        print('Evaluating dataset')

        total_loss_joint = 0.0
        total_loss_classification = 0.0
        total_loss_regression = 0.0

        for iter_num, data in enumerate(dataloader_val):

            if iter_num % 100 == 0:
                print('{}/{}'.format(iter_num, len(dataset_val)))

            with torch.no_grad():
                classification, regression, anchors = retinanet(data['img'].cuda().float())
                classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

                total_loss_joint += float(classification_loss + regression_loss)
                total_loss_regression += float(regression_loss)
                total_loss_classification += float(classification_loss)

        total_loss_joint /= (float(len(dataset_val)) / parser.batch_size)
        total_loss_classification /= (float(len(dataset_val)) / parser.batch_size)
        total_loss_regression /= (float(len(dataset_val)) / parser.batch_size)

        print('Validation epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Total loss: {:1.5f}'.format(epoch_num, float(total_loss_classification), float(total_loss_regression), float(total_loss_joint)))

    torch.save(retinanet.state_dict(), '{}{}_retinanet_{}.state_dict'.format(parser.dump_prefix, parser.dataset, epoch_num))

retinanet.eval()
torch.save(retinanet.state_dict(), '{}model_final.state_dict'.format(parser.dump_prefix, epoch_num))

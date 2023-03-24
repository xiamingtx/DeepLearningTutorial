#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:resnet_fruit_veg_classification.py
# author:xm
# datetime:2023/3/24 17:57
# software: PyCharm

"""
基于ResNet18的果蔬分类
数据集: https://aistudio.baidu.com/aistudio/datasetdetail/119023
"""

# import module your need

import math
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import random
import shutil
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import timm
from timm.utils import accuracy
from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from typing import Iterable
import sys

use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')


def get_args_parser():
    """获取参数"""
    parser = argparse.ArgumentParser('resnet classification', add_help=False)
    parser.add_argument('--batch_size', default=72, type=int,
                        help='Batch Size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch_size under memory constraints)')

    # Model parameters
    parser.add_argument('--input_size', default=128, type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', default=0.0001, type=float, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--root_path', default='dataset_fruit_veg', help='path where to save, empty for no saving')
    parser.add_argument('--output_dir', default='./output_dir_pretrained',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained', help='path where to tensorboard log')

    parser.add_argument('--resume', default='output_dir_pretrained/checkpoint-30.pth', help='resume from checkpoint')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient '
                                                               '(sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def split_dataset(test_split_ratio=0.05, desired_size=128, raw_path='.\\raw'):
    """
    对所有图片进行RGB转换 统一调整到大小一致 但不让图片发生变形或扭曲
    :param test_split_ratio:
    :param desired_size:
    :param raw_path:
    :return:
    tips: 这里在window下适用, 在Linux中 请将\\ 全部更换成 /
    """
    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print(f'Totally {len(dirs)} classes: {dirs}')

    for path in dirs:
        # 对每个类别单独处理
        path = path.split('\\')[-1]  # 类别

        os.makedirs(f'dataset_fruit_veg/train\\{path}', exist_ok=True)  # 在train中创建
        os.makedirs(f'dataset_fruit_veg/test\\{path}', exist_ok=True)  # 在test中创建

        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        # files += glob.glob(os.path.join(raw_path, path, '*.JPG'))  # 在windows下 *.jpg会都匹配上
        files += glob.glob(os.path.join(raw_path, path, '*.png'))

        random.shuffle(files)

        boundary = int(len(files) * test_split_ratio)  # 训练集和测试集的边界 boundary数目留给测试集

        for i, file in enumerate(files):
            img = Image.open(file).convert("RGB")
            old_size = img.size  # old_size[0] is in (width, height) format

            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = img.resize(new_size, Image.LANCZOS)

            new_im = Image.new('RGB', (desired_size, desired_size))
            new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(
                    os.path.join(f'dataset_fruit_veg/test\\{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(
                    os.path.join(f'dataset_fruit_veg/train\\{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))

    test_files = glob.glob(os.path.join('dataset_fruit_veg/test', '*', '*.jpg'))
    train_files = glob.glob(os.path.join('dataset_fruit_veg/train', '*', '*.jpg'))

    print(f'Totally {len(train_files)} files for training')
    print(f'Totally {len(test_files)} files for test')


split_dataset()


def statistic_mean_std():
    """
    统计数据库中所有图片 每个通道的均值和标准差
    :return:
    """
    train_files = glob.glob(os.path.join('dataset_fruit_veg/train', '*', '*.jpg'))
    print(f'Totally {len(train_files)} files for training')

    result = []
    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = img / 255.
        result.append(img)

    print(np.shape(result))  # [BS, H, W, C]
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)


statistic_mean_std()


def build_transform(is_train, args):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        print('train transform')
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size, args.input_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                torchvision.transforms.ToTensor()
            ]
        )
    # eval transform
    print('eval transform')
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor()
        ]
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset.find_classes(path)
    print(f'findng classes from {path}:\t{info[0]}')
    print(f'mappiing classes from {path} to indexes:\t{info[1]}')

    return dataset


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        output = nn.functional.softmax(output, dim=-1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: nn.Module, criterion: nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None, args=None):
    model.train(True)

    print_freq = 2

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(data_loader):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(samples)

        # warmup_lr = args.lr * min(1.0, epoch / 2.0)
        warmup_lr = args.lr
        optimizer.param_groups[0]['lr'] = warmup_lr

        loss = criterion(outputs, targets)
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss_value = loss.item()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print(f'Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss}, Lr: {warmup_lr}')


def main(args, mode='train', test_image_path=''):
    print(f'{mode} mode on {device}...')
    if mode == 'train':
        # 构建数据批次
        dataset_train = build_dataset(is_train=True, args=args)
        dataset_val = build_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # 构建模型
        model = timm.create_model('resnet18', pretrained=False, num_classes=36, drop_rate=0.1, drop_path_rate=0.1)
        model = model.to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        os.makedirs(args.log_dir, exist_ok=True)

        log_writer = SummaryWriter(log_dir=args.log_dir)

        loss_scaler = NativeScaler()

        # 读入已有的模型
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        for epoch in range(args.start_epoch, args.epochs):
            print(f'Epoch {epoch}')
            print(f'length of data_loader is {len(data_loader_train)}')

            if epoch % 1 == 0:
                print('Evaluating...')
                model.eval()
                test_stats = evaluate(data_loader_val, model, device)
                print(f"Accuracy of the network on the {len(data_loader_val)} test images: {test_stats['acc1']:.1f}%")

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                model.train()

            print('training')
            train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch + 1,
                            loss_scaler, None, log_writer=log_writer, args=args)
            if args.output_dir:
                print('Saving checkpoints...')
                misc.save_model(args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch)
    else:
        model = timm.create_model('resnet18', pretrained=False, num_classes=36, drop_rate=0.1, drop_path_rate=0.1)
        model = model.to(device)
        class_dict = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 'capsicum': 5,
                      'carrot': 6, 'cauliflower': 7, 'chilli pepper': 8, 'corn': 9, 'cucumber': 10, 'eggplant': 11,
                      'garlic': 12, 'ginger': 13, 'grapes': 14, 'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18,
                      'mango': 19, 'onion': 20, 'orange': 21, 'paprika': 22, 'pear': 23, 'peas': 24, 'pineapple': 25,
                      'pomegranate': 26, 'potato': 27, 'raddish': 28, 'soy beans': 29, 'spinach': 30, 'sweetcorn': 31,
                      'sweetpotato': 32, 'tomato': 33, 'turnip': 34, 'watermelon': 35}

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of trainable params (M): %.2f' % (n_parameters / 1.e6))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        os.makedirs(args.log_dir, exist_ok=True)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        model.eval()

        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((args.input_size, args.input_size), Image.LANCZOS)
        image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)

        output = nn.functional.softmax(output, dim=-1)
        class_idx = torch.argmax(output, dim=1)[0]
        score = torch.max(output, dim=1)[0][0]

        print(f'image path is {test_image_path}')
        print(f'score is {score.item()}, class id is {class_idx.item()}, class name is '
              f'{list(class_dict.keys())[list(class_dict.values()).index(class_idx)]}')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mode = 'train'  # infer or train

    if mode == 'train':
        main(args, mode=mode)
    else:
        images = glob.glob('dataset_fruit_veg/test/*/*.jpg')  # 仅做测试 改成你的路径
        # random.shuffle(images)

        for image in images:
            print('\n')
            main(args, mode=mode, test_image_path=image)

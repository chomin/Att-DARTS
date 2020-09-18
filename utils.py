import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import Module

from genotypes import Genotype


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """

    :param output: logits, [b, classes]
    :param target: [b]
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    """

    :param args:
    :return:
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    """

    :param args:
    :return:
    """
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model: Module) -> float:
    """
    count all parameters excluding auxiliary
    :param model:
    :return:
    """

    return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save_path):
    filename = os.path.join(save_path, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print('saved to model:', model_path)
    torch.save(model.state_dict(), model_path)


def load(model, saved_data, to_parallel):
    print('load from model:', saved_data)

    def _fix_model_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if to_parallel:
                if not name.startswith('module.'):
                    name = 'module.' + name
            else:
                if name.startswith('module.'):
                    name = name[7:]  # remove 'module.' of dataparallel

            name.replace('ops', 'first_layers')
            name.replace('attns', 'second_layers')
            new_state_dict[name] = v
        return new_state_dict

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if isinstance(saved_data, str) or isinstance(saved_data, Path):
        dic = torch.load(saved_data, map_location=device)
        if 'state_dict' in dic:
            dic = dic['state_dict']

    elif isinstance(saved_data, OrderedDict):
        dic = saved_data
    else:
        raise Exception(f'saved_data must be model_path or state_dict. found type: {type(saved_data)}')

    dic = _fix_model_state_dict(dic)
    model.load_state_dict(dic)


def save_genotype(genotype, path):
    with open(path, 'w') as f:
        dic = genotype._asdict()
        if isinstance(dic['normal_concat'], range):
            dic['normal_concat'] = [i for i in dic['normal_concat']]
        if isinstance(dic['reduce_concat'], range):
            dic['reduce_concat'] = [i for i in dic['reduce_concat']]

        json.dump(dic, f, indent=4)

    print(f'Saved genotype at {path}\n'
          f'genotype: {genotype}')


def save_genotypes(genotypes_with_paths: List[Tuple[Genotype, str]]):
    for genotype, path in genotypes_with_paths:
        save_genotype(genotype, path)


def load_genotype(path):
    with open(path) as f:
        dic = json.load(f)
        normal = [(i[0], i[1], i[2]) for i in dic['normal']]
        reduce = [(i[0], i[1], i[2]) for i in dic['reduce']]
        normal_bottleneck = dic['normal_bottleneck'] if 'normal_bottleneck' in dic else ''
        reduce_bottleneck = dic['reduce_bottleneck'] if 'reduce_bottleneck' in dic else ''

        g = Genotype(normal=normal, normal_concat=dic['normal_concat'], reduce=reduce,
                     reduce_concat=dic['reduce_concat'],
                     normal_bottleneck=normal_bottleneck, reduce_bottleneck=reduce_bottleneck)
        return g


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path: Path, scripts_to_save=None):
    path.mkdir(parents=True, exist_ok=True)
    print(f'Experiment dir : {path}')

    path.mkdir(parents=True, exist_ok=True)

    if scripts_to_save is not None:
        scripts_dir = path / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)

        for script in scripts_to_save:
            dst_file = scripts_dir / os.path.basename(script)
            shutil.copyfile(script, dst_file)

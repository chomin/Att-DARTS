import logging
import os
import sys
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

import utils
from constants import MyDataset
from genotypes import Genotype
from model import NetworkCIFAR, NetworkImageNet

use_DataParallel = torch.cuda.device_count() > 1
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class Trainer:

    def __init__(self, args: Namespace, genotype: Genotype, my_dataset: MyDataset, choose_cell=False):

        self.__args = args
        self.__dataset = my_dataset
        self.__previous_epochs = 0

        if args.seed is None:
            raise Exception('designate seed.')
        elif args.epochs is None:
            raise Exception('designate epochs.')
        if not (args.arch or args.arch_path):
            raise Exception('need to designate arch.')

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        np.random.seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.manual_seed(args.seed)

        logging.info(f'gpu device = {args.gpu}')
        logging.info(f'args = {args}')

        logging.info(f'Train genotype: {genotype}')

        if my_dataset == MyDataset.CIFAR10:
            self.model = NetworkCIFAR(args.init_ch, 10, args.layers, args.auxiliary, genotype)
            train_transform, valid_transform = utils._data_transforms_cifar10(args)
            train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)


        elif my_dataset == MyDataset.CIFAR100:
            self.model = NetworkCIFAR(args.init_ch, 100, args.layers, args.auxiliary, genotype)
            train_transform, valid_transform = utils._data_transforms_cifar100(args)
            train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

        elif my_dataset == MyDataset.ImageNet:
            self.model = NetworkImageNet(args.init_ch, 1000, args.layers, args.auxiliary, genotype)
            self.__criterion_smooth = CrossEntropyLabelSmooth(1000, args.label_smooth).to(device)
            traindir = os.path.join(args.data, 'train')
            validdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_data = dset.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.2),
                    transforms.ToTensor(),
                    normalize,
                ]))
            valid_data = dset.ImageFolder(
                validdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
        else:
            raise Exception('No match Dataset')

        checkpoint = None
        if use_DataParallel:
            print('use Data Parallel')
            if args.checkpoint_path:
                checkpoint = torch.load(args.checkpoint_path)
                utils.load(self.model, checkpoint['state_dict'], args.to_parallel)
                self.__previous_epochs = checkpoint['epoch']
                args.epochs -= self.__previous_epochs
                if args.epochs <= 0:
                    raise Exception('args.epochs is too small.')

            self.model = nn.DataParallel(self.model)
            self.__module = self.model.module
            torch.cuda.manual_seed_all(args.seed)
        else:
            if args.checkpoint_path:
                checkpoint = torch.load(args.checkpoint_path)
                utils.load(self.model, checkpoint['state_dict'], args.to_parallel)
                args.epochs -= checkpoint['epoch']
                if args.epochs <= 0:
                    raise Exception('args.epochs is too small.')
            torch.cuda.manual_seed(args.seed)
            self.__module = self.model

        self.model.to(device)

        param_size = utils.count_parameters_in_MB(self.model)
        logging.info(f'param size = {param_size}MB')

        self.__criterion = nn.CrossEntropyLoss().to(device)

        self.__optimizer = torch.optim.SGD(
            self.__module.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd
        )
        if checkpoint:
            self.__optimizer.load_state_dict(checkpoint['optimizer'])

        num_workers = torch.cuda.device_count() * 4
        if choose_cell:
            num_train = len(train_data)  # 50000
            indices = list(range(num_train))
            split = int(np.floor(args.train_portion * num_train))  # 25000

            self.__train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=args.batchsz,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                pin_memory=True, num_workers=num_workers)

            self.__valid_queue = torch.utils.data.DataLoader(
                train_data, batch_size=args.batchsz,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
                pin_memory=True, num_workers=num_workers)
        else:
            self.__train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=num_workers)

            self.__valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=num_workers)

        if my_dataset == MyDataset.CIFAR10 or MyDataset.CIFAR100:
            self.__scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.__optimizer, args.epochs)
        elif my_dataset == MyDataset.ImageNet:
            self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, args.decay_period, gamma=args.gamma)
        else:
            raise Exception('No match Dataset')

        if checkpoint:
            self.__scheduler.load_state_dict(checkpoint['scheduler'])

    def __train_epoch(self, train_queue, model, criterion, optimizer, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.train()

        with tqdm(train_queue) as progress:
            progress.set_description_str(f'Train epoch {epoch}')

            for step, (x, target) in enumerate(progress):

                x, target = x.to(device), target.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits, logits_aux = model(x)
                loss = criterion(logits, target)
                if self.__args.auxiliary:
                    loss_aux = criterion(logits_aux, target)
                    loss += self.__args.auxiliary_weight * loss_aux
                loss.backward()
                nn.utils.clip_grad_norm_(model.module.parameters() if use_DataParallel else model.parameters(),
                                         self.__args.grad_clip)
                optimizer.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = x.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                progress.set_postfix_str(f'loss: {objs.avg}, top1: {top1.avg}')

                if step % self.__args.report_freq == 0:
                    logging.info(f'Step:{step:03} loss:{objs.avg} acc1:{top1.avg} acc5:{top5.avg}')

        return top1.avg, objs.avg

    def __infer_epoch(self, valid_queue, model, criterion, epoch):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.eval()

        with tqdm(valid_queue) as progress:
            for step, (x, target) in enumerate(progress):
                progress.set_description_str(f'Valid epoch {epoch}')

                x = x.to(device)
                target = target.to(device, non_blocking=True)

                with torch.no_grad():
                    logits, _ = model(x)
                    loss = criterion(logits, target)

                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    n = x.size(0)
                    objs.update(loss.item(), n)
                    top1.update(prec1.item(), n)
                    top5.update(prec5.item(), n)

                progress.set_postfix_str(f'loss: {objs.avg}, top1: {top1.avg}')

                if step % self.__args.report_freq == 0:
                    logging.info(f'>> Validation: {step:03} {objs.avg} {top1.avg} {top5.avg}')

        return top1.avg, top5.avg, objs.avg

    def train(self) -> Tuple[float, float, float]:

        best_acc_top1 = 0
        for epoch in tqdm(range(self.__args.epochs), desc='Total Progress'):
            self.__scheduler.step()
            logging.info(f'epoch {epoch} lr {self.__scheduler.get_lr()[0]}')
            self.__module.drop_path_prob = self.__args.drop_path_prob * epoch / self.__args.epochs

            train_acc, train_obj = self.__train_epoch(self.__train_queue, self.model, self.__criterion,
                                                      self.__optimizer, epoch + 1)
            logging.info(f'train_acc: {train_acc}')

            valid_acc_top1, valid_acc_top5, valid_obj = self.__infer_epoch(self.__valid_queue, self.model,
                                                                           self.__criterion, epoch + 1)
            logging.info(f'valid_acc: {valid_acc_top1}')
            if self.__dataset == MyDataset.ImageNet:
                logging.info(f'valid_acc_top5 {valid_acc_top5}')

            is_best = False
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True

            utils.save(self.model, os.path.join(self.__args.save, 'trained.pt'))
            utils.save_checkpoint({
                'epoch': epoch + 1 + self.__previous_epochs,
                'state_dict': self.model.state_dict(),
                'optimizer': self.__optimizer.state_dict(),
                'scheduler': self.__scheduler.state_dict()
            }, is_best=is_best, save_path=self.__args.save)
            print('saved to: trained.pt and checkpoint.pth.tar')

        return train_acc, valid_acc_top1, best_acc_top1

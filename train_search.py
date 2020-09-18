import argparse
import glob
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torch import optim
from tqdm import tqdm

import utils
from arch import Arch
from constants import DATA_DIRECTORY
from model_search import Network, AttLocation

parser = argparse.ArgumentParser('cifar')
parser.add_argument('--data', type=Path, default=DATA_DIRECTORY, help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=str, default='', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--checkpoint_path', type=Path, help='path to checkpoint for restart')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_len', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--exp_path', type=Path, default='search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--location', type=AttLocation, default=AttLocation.AFTER_EVERY,
                    choices=list(AttLocation))
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
use_DataParallel = torch.cuda.device_count() > 1


def main():
    args.exp_path /= f'{args.gpu}_{time.strftime("%Y%m%d-%H%M%S")}'
    utils.create_exp_dir(Path(args.exp_path), scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(args.exp_path / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.seed is None:
        raise Exception('designate seed.')
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.manual_seed(args.seed)

    # ================================================
    # total, used = os.popen(
    #     'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    # ).read().split('\n')[args.gpu].split(',')
    # total = int(total)
    # used = int(used)

    # print('Total GPU mem:', total, 'used:', used)

    # try:
    #     block_mem = 0.85 * (total - used)
    #     print(block_mem)
    #     x = torch.empty((256, 1024, int(block_mem))).cuda()
    #     del x
    # except RuntimeError as err:
    #     print(err)
    #     block_mem = 0.8 * (total - used)
    #     print(block_mem)
    #     x = torch.empty((256, 1024, int(block_mem))).cuda()
    #     del x
    #
    #
    # print('reuse mem now ...')
    # ================================================

    logging.info(f'GPU device = {args.gpu}')
    logging.info(f'args = {args}')

    criterion = nn.CrossEntropyLoss().to(device)

    setting = args.location

    model = Network(args.init_ch, 10, args.layers, criterion, setting)
    checkpoint = None
    previous_epochs = 0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        utils.load(model, checkpoint['state_dict'], False)
        previous_epochs = checkpoint['epoch']
        args.epochs -= previous_epochs
        if args.epochs <= 0:
            raise Exception('args.epochs is too small.')

    if use_DataParallel:
        print('use Data Parallel')
        model = nn.parallel.DataParallel(model)
        model = model.cuda()
        module = model.module
        torch.cuda.manual_seed_all(args.seed)
    else:
        model = model.to(device)
        module = model

    param_size = utils.count_parameters_in_MB(model)
    logging.info(f'param size = {param_size}MB')

    arch_and_attn_params = list(
        map(id, module.arch_and_attn_parameters() if use_DataParallel else model.arch_and_attn_parameters()))
    weight_params = filter(lambda p: id(p) not in arch_and_attn_params,
                           module.parameters() if use_DataParallel else model.parameters())

    optimizer = optim.SGD(weight_params, args.lr, momentum=args.momentum, weight_decay=args.wd)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)  # 50000
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))  # 25000

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=8)  # from 2

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batchsz,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
        pin_memory=True, num_workers=8)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr_min)
    if checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    arch = Arch(model, criterion, args)
    if checkpoint:
        arch.optimizer.load_state_dict(checkpoint['arch_optimizer'])

    for epoch in tqdm(range(args.epochs), desc='Total Progress'):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        logging.info(f'\nEpoch: {epoch} lr: {lr}')
        gen = module.genotype()
        logging.info(f'Genotype: {gen}')

        print(F.softmax(module.alphas_normal, dim=-1))
        print(F.softmax(module.alphas_reduce, dim=-1))
        if module.betas_normal is not None:
            print(F.softmax(module.betas_normal, dim=-1))
            print(F.softmax(module.betas_reduce, dim=-1))
        if module.gammas_normal is not None:
            print(F.softmax(module.gammas_normal, dim=-1))
            print(F.softmax(module.gammas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, arch, criterion, optimizer, lr, epoch + 1)
        logging.info(f'train acc: {train_acc}')

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch + 1)
        logging.info(f'valid acc: {valid_acc}')

        utils.save(model, args.exp_path / 'search.pt')
        utils.save_checkpoint({
            'epoch': epoch + 1 + previous_epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'arch_optimizer': arch.optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, False, args.exp_path)

        gen = module.genotype()
        gen_path = args.exp_path / 'genotype.json'
        utils.save_genotype(gen, gen_path)

        logging.info(f'Result genotype: {gen}')


def train(train_queue, valid_queue, model, arch, criterion, optimizer, lr, epoch):
    """

    :param train_queue: train loader
    :param valid_queue: validate loader
    :param model: network
    :param arch: Arch class
    :param criterion:
    :param optimizer:
    :param lr:
    :param epoch:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    valid_iter = iter(valid_queue)

    with tqdm(train_queue) as progress:
        for step, (x, target) in enumerate(progress):
            progress.set_description_str(f'Train epoch {epoch}')
            batchsz = x.size(0)
            model.train()

            # [b, 3, 32, 32], [40]
            x, target = x.to(device), target.to(device, non_blocking=True)
            x_search, target_search = next(valid_iter)  # [b, 3, 32, 32], [b]
            x_search, target_search = x_search.to(device), target_search.to(device, non_blocking=True)

            # 1. update alpha
            arch.step(x, target, x_search, target_search, lr, optimizer, unrolled=args.unrolled)

            logits = model(x)
            loss = criterion(logits, target)

            # 2. update weight
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters() if use_DataParallel else model.parameters(),
                                     args.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            losses.update(loss.item(), batchsz)
            top1.update(prec1.item(), batchsz)
            top5.update(prec5.item(), batchsz)

            progress.set_postfix_str(f'loss: {losses.avg}, top1: {top1.avg}')

            if step % args.report_freq == 0:
                logging.info(f'Step:{step:03} loss:{losses.avg} acc1:{top1.avg} acc5:{top5.avg}')

    return top1.avg, losses.avg


def infer(valid_queue, model, criterion, epoch):
    """

    :param valid_queue:
    :param model:
    :param criterion:
    :return:
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        with tqdm(valid_queue) as progress:
            for step, (x, target) in enumerate(progress):

                progress.set_description_str(f'Valid epoch {epoch}')

                x, target = x.to(device), target.cuda(non_blocking=True)
                batchsz = x.size(0)

                logits = model(x)
                loss = criterion(logits, target)

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                losses.update(loss.item(), batchsz)
                top1.update(prec1.item(), batchsz)
                top5.update(prec5.item(), batchsz)

                progress.set_postfix_str(f'loss: {losses.avg}, top1: {top1.avg}')

                if step % args.report_freq == 0:
                    logging.info(f'>> Validation: {step:03} {losses.avg} {top1.avg} {top5.avg}')

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()

import argparse
import glob
import logging
import socket
import sys
import time
from pathlib import Path

import torch.utils
from torch.nn import DataParallel

import utils
import genotypes
from constants import DATA_DIRECTORY, MyDataset
from tester import Tester
from trainer import Trainer


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument('--data', type=Path, default=DATA_DIRECTORY / 'imagenet', help='location of the data corpus')
    parser.add_argument('--batchsz', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=3e-5, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gpu', type=str, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
    parser.add_argument('--init_ch', type=int, default=48, help='num of init channels')
    parser.add_argument('--layers', type=int, default=14, help='total number of layers')
    parser.add_argument('--checkpoint_path', type=Path, help='path to checkpoint for restart')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    parser.add_argument('--exp_path', type=Path, default=Path('exp_imagenet'), help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='', help='which architecture to use')
    parser.add_argument('--arch_path', type=str, default='', help='which architecture of json to use')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    args = parser.parse_args()

    my_dataset = MyDataset.ImageNet
    args.save = args.exp_path / f'ImageNet-{time.strftime("%Y%m%d-%H%M%S")}'
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    genotype = eval(f'genotypes.{args.arch}') if not args.arch_path else utils.load_genotype(args.arch_path)
    trainer = Trainer(args, genotype, my_dataset)
    _, _, _ = trainer.train()

    args.seed = 0
    test_model = trainer.model.module if isinstance(trainer.model, DataParallel) else trainer.model
    tester = Tester(test_args=args, my_dataset=my_dataset, model=test_model)
    valid_acc_top1, valid_acc_top5, valid_obj = tester.infer()
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)
    logging.info('valid_err_top1 %f', 100 - valid_acc_top1)
    logging.info('valid_err_top5 %f', 100 - valid_acc_top5)


if __name__ == '__main__':
    main()

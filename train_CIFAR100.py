import argparse
import glob
import logging
import time
from pathlib import Path

from torch.nn import DataParallel

import utils
import genotypes
from constants import DATA_DIRECTORY, MyDataset
from tester import Tester
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser('cifar100')
    parser.add_argument('--data', type=Path, default=DATA_DIRECTORY, help='location of the data corpus')
    parser.add_argument('--batchsz', type=int, default=96, help='batch size')
    parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gpu', type=str, default='', help='gpu device id')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--checkpoint_path', type=Path, help='path to checkpoint for restart')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--exp_path', type=Path, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--arch', type=str, default='', help='which architecture to use')
    parser.add_argument('--arch_path', type=str, default='', help='which architecture of json to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    args = parser.parse_args()

    args.save = args.exp_path / f'CIFAR100-{time.strftime("%Y%m%d-%H%M%S")}'
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    # ----- train -----
    my_dataset = MyDataset.CIFAR100
    genotype = eval(f'genotypes.{args.arch}') if not args.arch_path else utils.load_genotype(args.arch_path)
    trainer = Trainer(args, genotype, my_dataset)
    _, _, _ = trainer.train()

    test_model = trainer.model.module if isinstance(trainer.model, DataParallel) else trainer.model
    tester = Tester(test_args=args, my_dataset=my_dataset, model=test_model)
    test_acc, _, test_obj = tester.infer()
    logging.info(f'test_acc {test_acc}')
    logging.info(f'test_error: {100 - test_acc}')


if __name__ == '__main__':
    main()

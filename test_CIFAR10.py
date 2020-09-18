import argparse
import logging
from pathlib import Path

from constants import DATA_DIRECTORY, MyDataset
from tester import Tester


def main():
    parser = argparse.ArgumentParser('cifar10')
    parser.add_argument('--data', type=Path, default=DATA_DIRECTORY, help='location of the data corpus')
    parser.add_argument('--batchsz', type=int, default=36, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=str, default='', help='gpu device id')
    parser.add_argument('--init_ch', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--model_path', type=Path, help='path of pretrained model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='', help='which architecture to use')
    parser.add_argument('--arch_path', type=str, default='', help='which architecture of json to use')
    args = parser.parse_args()

    tester = Tester(test_args=args, my_dataset=MyDataset.CIFAR10)
    test_acc, _, test_obj = tester.infer()
    logging.info(f'test_acc {test_acc}')
    logging.info(f'test_error: {100 - test_acc}')


if __name__ == '__main__':
    main()

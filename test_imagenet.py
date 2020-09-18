import argparse
import logging
from pathlib import Path

from constants import DATA_DIRECTORY, MyDataset
from tester import Tester


def main():
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument('--data', type=Path, default=DATA_DIRECTORY / 'imagenet',
                        help='location of the data corpus')
    parser.add_argument('--batchsz', type=int, default=128, help='batch size')
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--gpu', type=str, help='gpu device id')
    parser.add_argument('--init_ch', type=int, default=48, help='num of init channels')
    parser.add_argument('--layers', type=int, default=14, help='total number of layers')
    parser.add_argument('--model_path', type=Path, help='path of pretrained model')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='', help='which architecture to use')
    parser.add_argument('--arch_path', type=str, default='', help='which architecture of json to use')
    args = parser.parse_args()

    tester = Tester(test_args=args, my_dataset=MyDataset.ImageNet)
    valid_acc_top1, valid_acc_top5, valid_obj = tester.infer()
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)
    logging.info('valid_err_top1 %f', 100 - valid_acc_top1)
    logging.info('valid_err_top5 %f', 100 - valid_acc_top5)


if __name__ == '__main__':
    main()

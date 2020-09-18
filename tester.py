import logging
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm

import utils
import genotypes
from model import NetworkCIFAR, NetworkImageNet
from constants import MyDataset


class Tester:
    def __init__(self, test_args: Namespace, my_dataset: MyDataset, model: nn.Module = None):

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        np.random.seed(test_args.seed)
        torch.manual_seed(test_args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True

        logging.info(f'gpu device = {test_args.gpu}')
        logging.info(f'args = {test_args}')

        if model is None:
            # equal to: genotype = genotypes.DARTS_v2
            if not (test_args.arch or test_args.arch_path):
                logging.info('need to designate arch.')
                sys.exit(1)

            genotype = eval(f'genotypes.{test_args.arch}') if not test_args.arch_path else utils.load_genotype(
                test_args.arch_path)
            print('Load genotype:', genotype)

            if my_dataset is MyDataset.CIFAR10:
                model = NetworkCIFAR(test_args.init_ch, 10, test_args.layers, test_args.auxiliary, genotype).to(
                    self.__device)
            elif my_dataset is MyDataset.CIFAR100:
                model = NetworkCIFAR(test_args.init_ch, 100, test_args.layers, test_args.auxiliary, genotype).to(
                    self.__device)
            elif my_dataset is MyDataset.ImageNet:
                model = NetworkImageNet(test_args.init_ch, 1000, test_args.layers, test_args.auxiliary, genotype).to(
                    self.__device)
            else:
                raise Exception('No match MyDataset')

            utils.load(model, test_args.model_path, False)
            model = model.to(self.__device)

            param_size = utils.count_parameters_in_MB(model)
            logging.info(f'param size = {param_size}MB')

        model.drop_path_prob = test_args.drop_path_prob
        self.__model = model

        self.__args = test_args
        self.__criterion = nn.CrossEntropyLoss().to(self.__device)

        if my_dataset is MyDataset.CIFAR10:
            _, test_transform = utils._data_transforms_cifar10(test_args)
            test_data = dset.CIFAR10(root=test_args.data, train=False, download=True, transform=test_transform)

        elif my_dataset is MyDataset.CIFAR100:
            _, test_transform = utils._data_transforms_cifar100(test_args)
            test_data = dset.CIFAR100(root=test_args.data, train=False, download=True, transform=test_transform)

        elif my_dataset is MyDataset.ImageNet:
            validdir = test_args.data / 'val'
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            valid_data = dset.ImageFolder(
                validdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            test_data = valid_data
        else:
            raise Exception('No match MyDataset')

        self.__test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=test_args.batchsz, shuffle=False, pin_memory=True, num_workers=4)

    def infer(self):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        self.__model.eval()

        with torch.no_grad():
            with tqdm(self.__test_queue) as progress:
                for step, (x, target) in enumerate(progress):
                    progress.set_description_str(f'Test: ')

                    x, target = x.to(self.__device), target.to(self.__device, non_blocking=True)

                    logits, _ = self.__model(x)
                    loss = self.__criterion(logits, target)

                    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                    batchsz = x.size(0)
                    objs.update(loss.item(), batchsz)
                    top1.update(prec1.item(), batchsz)
                    top5.update(prec5.item(), batchsz)

                    progress.set_postfix_str(f'loss: {objs.avg}, top1: {top1.avg}')

                    if step % self.__args.report_freq == 0:
                        logging.info(f'test {step:03} {objs.avg} {top1.avg} {top5.avg}')

        return top1.avg, top5.avg, objs.avg

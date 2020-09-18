from attentions import ATTNS
from operations import FactorizedReduce, ReLUConvBN, OPS, Identity
from utils import drop_path

import torch
import torch.nn as nn


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, height, width):
        """

        :param genotype:
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction:
        :param reduction_prev:
        """
        super(Cell, self).__init__()

        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            first_layers, indices, second_layers = zip(*genotype.reduce)
            concat = genotype.reduce_concat
            bottleneck = genotype.reduce_bottleneck
        else:
            first_layers, indices, second_layers = zip(*genotype.normal)
            concat = genotype.normal_concat
            bottleneck = genotype.normal_bottleneck
        self._compile(C, first_layers, second_layers, indices, concat, reduction, bottleneck, height, width)

    def _compile(self, C, first_layers, second_layers, indices, concat, reduction, bottleneck, height, width):
        """

        :param C:
        :param first_layers:
        :param indices:
        :param concat:
        :param reduction:
        :return:
        """
        assert len(first_layers) == len(indices)

        self._steps = len(first_layers) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._first_layers = nn.ModuleList()
        self._second_layers = nn.ModuleList()
        for first_layer_name, second_layer_name, index in zip(first_layers, second_layers, indices):
            stride = 2 if reduction and index < 2 else 1
            if first_layer_name in OPS:
                first_layer = OPS[first_layer_name](C, stride, True)
            elif reduction:  # for mixed
                first_layer = nn.Sequential(OPS['skip_connect'](C, stride, False),
                                            ATTNS[first_layer_name](C, height, width))
            else:
                first_layer = ATTNS[first_layer_name](C, height, width)
            self._first_layers += [first_layer]

            if second_layer_name:
                stride = 1
                if second_layer_name in OPS:
                    second_layer = OPS[second_layer_name](C, stride, True)
                else:
                    second_layer = ATTNS[second_layer_name](C, height, width)
                self._second_layers += [second_layer]

        self._bottleneck = None
        if bottleneck:
            self._bottleneck = ATTNS[bottleneck](C * self.multiplier, height, width)

        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        """

        :param s0:
        :param s1:
        :param drop_prob:
        :return:
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._first_layers[2 * i]
            op2 = self._first_layers[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self._second_layers:
                at1 = self._second_layers[2 * i]
                at2 = self._second_layers[2 * i + 1]
                h1 = at1(h1)
                h2 = at2(h2)

                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity) and not isinstance(at1, Identity):
                        h1 = drop_path(h1, drop_prob)
                    if not isinstance(op2, Identity) and not isinstance(at2, Identity):
                        h2 = drop_path(h2, drop_prob)
            else:
                if self.training and drop_prob > 0.:
                    if not isinstance(op1, Identity):
                        h1 = drop_path(h1, drop_prob)
                    if not isinstance(op2, Identity):
                        h2 = drop_path(h2, drop_prob)

            s = h1 + h2
            states += [s]

        out = torch.cat([states[i] for i in self._concat], dim=1)
        if self._bottleneck:
            out = self._bottleneck(out)
        return out


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()

        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        height_curr = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                height_curr //= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, height_curr, height_curr)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class NetworkImageNet(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C
        height_curr = 128

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                height_curr //= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, height_curr, height_curr)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux

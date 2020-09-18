from typing import Tuple
from enum import Enum, unique

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from operations import OPS, FactorizedReduce, ReLUConvBN
from attentions import ATTNS
from genotypes import PRIMITIVES, ATTN_PRIMIVIVES, Genotype

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


@unique
class AttLocation(Enum):
    AFTER_EVERY = 'after_every'
    END = 'end'
    AFTER_EVERY_AND_END = 'after_every_and_end'
    MIXED_WITH_OPERATION = 'mixed_with_operation'
    DOUBLE_MIXED = 'double_mixed'
    NO_ATTENTION = 'no_attention'

    def __str__(self):
        return self.value


class MixedLayer(nn.Module):
    """
    a mixtures output of 8 type of units.
    we use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """

    def __init__(self, c: int, stride: int, height: int, width: int, setting: AttLocation):
        """
        :param c: channel
        :param stride: stride of convolutions
        """
        super(MixedLayer, self).__init__()

        self.first_layers = nn.ModuleList()
        self.second_layers = nn.ModuleList()
        self.setting = setting

        def _make_operation_layers(is_reduce_layer: bool) -> nn.ModuleList:
            layers = nn.ModuleList()
            for primitive in PRIMITIVES:
                # create corresponding layer
                layer = OPS[primitive](c, stride if is_reduce_layer else 1, False)

                # append batchnorm after pool layer
                if 'pool' in primitive:
                    # disable affine w/b for batchnorm
                    layer = nn.Sequential(layer, nn.BatchNorm2d(c, affine=False))
                layers.append(layer)
            return layers

        def _make_mixed_layers(is_reduce_layer: bool) -> nn.ModuleList:
            layers = _make_operation_layers(is_reduce_layer)
            for primitive in ATTN_PRIMIVIVES[1:]:
                layer = ATTNS[primitive](c, height, width)
                if stride != 1:
                    layer = nn.Sequential(OPS['skip_connect'](c, stride if is_reduce_layer else 1, False), layer)
                layers.append(layer)

            return layers

        # ---------- make first_layers ----------
        if setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.DOUBLE_MIXED]:
            self.first_layers = _make_mixed_layers(is_reduce_layer=True)

        elif setting in [AttLocation.AFTER_EVERY, AttLocation.NO_ATTENTION, AttLocation.END, AttLocation.AFTER_EVERY_AND_END]:
            self.first_layers = _make_operation_layers(is_reduce_layer=True)

        else:
            raise Exception("no match setting")

        # ---------- make second_layers ----------
        if setting in [AttLocation.AFTER_EVERY, AttLocation.AFTER_EVERY_AND_END]:
            for attn_primitive in ATTN_PRIMIVIVES:
                attn = ATTNS[attn_primitive](c, height, width)
                self.second_layers.append(attn)

        elif setting is AttLocation.DOUBLE_MIXED:
            self.second_layers = _make_mixed_layers(is_reduce_layer=False)

        elif setting in [AttLocation.NO_ATTENTION, AttLocation.END, AttLocation.MIXED_WITH_OPERATION]:
            pass

        else:
            raise Exception("no match setting")

    def forward(self, x, weights_a, weights_b):
        """
        :param weights_b:
        :param x: data
        :param weights_a: alpha,[op_num:8], the output = sum of alpha * op(x)
        :return:
        """

        assert len(weights_a) == len(
            self.first_layers), f'len(weights_a): {len(weights_a)}, len(self.first_layers): {len(self.first_layers)}'

        if weights_b is not None:
            assert len(weights_b) == len(
                self.second_layers), f'len(weights_b): {len(weights_b)}, len(self.second_layers): {len(self.second_layers)}'

        h = [w * layer(x) for w, layer in zip(weights_a, self.first_layers)]
        # element-wise add by torch.add
        h = sum(h)

        if self.setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.END, AttLocation.NO_ATTENTION]:
            return h

        elif self.setting in [AttLocation.AFTER_EVERY, AttLocation.AFTER_EVERY_AND_END, AttLocation.DOUBLE_MIXED]:
            out = [w_b * attn(h) for w_b, attn in zip(weights_b, self.second_layers)]
            out = sum(out)
            return out

        else:
            raise Exception("no match setting")


class Cell(nn.Module):

    def __init__(self, steps: int, multiplier: int, cpp: int, cp: int, c: int, reduction: bool, reduction_prev: bool,
                 height: int, width: int, setting: AttLocation):
        """
        :param steps: 4, number of layers inside a cell
        :param multiplier: 4
        :param cpp: 48
        :param cp: 48
        :param c: 16
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
        super(Cell, self).__init__()

        # indicating current cell is reduction or not
        self.reduction = reduction
        self.reduction_prev = reduction_prev

        self.setting = setting

        # preprocess0 deal with output from prev_prev cell
        if reduction_prev:
            # if prev cell has reduced channel/double width,
            # it will reduce width by half
            self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
        # preprocess1 deal with output from prev cell
        self.preprocess1 = ReLUConvBN(cp, c, 1, 1, 0, affine=False)

        # steps inside a cell
        self.steps = steps  # 4
        self.multiplier = multiplier  # 4

        self.layers = nn.ModuleList()

        for i in range(self.steps):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if reduction and j < 2 else 1
                layer = MixedLayer(c, stride, height, width, setting)
                self.layers.append(layer)

        self.bottleneck_attns = nn.ModuleList()
        if setting in [AttLocation.END, AttLocation.AFTER_EVERY_AND_END]:
            for attn_primitive in ATTN_PRIMIVIVES:
                attn = ATTNS[attn_primitive](c * steps, height, width)
                self.bottleneck_attns.append(attn)

        elif setting in [AttLocation.AFTER_EVERY, AttLocation.NO_ATTENTION, AttLocation.MIXED_WITH_OPERATION, AttLocation.DOUBLE_MIXED]:
            pass

        else:
            raise Exception('no match setting')

    def forward(self, s0, s1, weights_a, weights_b, weights_c):
        """
        :param s0:
        :param s1:
        :param weights_a: [14, 8]
        :param weights_b: weights for attentions in each edge
        :param weights_c: weights for attentions in bottleneck
        :return:
        """
        # print('s0:', s0.shape,end='=>')
        s0 = self.preprocess0(s0)  # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s0.shape, self.reduction_prev)
        # print('s1:', s1.shape,end='=>')
        s1 = self.preprocess1(s1)  # [40, 48, 32, 32], [40, 16, 32, 32]
        # print(s1.shape)

        states = [s0, s1]
        offset = 0
        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.steps):  # 4 intermediate nodes
            # [40, 16, 32, 32]
            s = sum(self.layers[offset + j]
                    (h, weights_a[offset + j], weights_b[offset + j] if weights_b is not None else None)
                    for j, h in enumerate(states))
            offset += len(states)
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            # print('node:',i, s.shape, self.reduction)

        # concat along dim=channel
        h = torch.cat(states[-self.multiplier:], dim=1)  # 6 of [40, 16, 32, 32]
        if self.setting in [AttLocation.END, AttLocation.AFTER_EVERY_AND_END]:

            out = [w * layer(h) for w, layer in zip(weights_c, self.bottleneck_attns)]
            out = sum(out)
            return out

        elif self.setting in [AttLocation.AFTER_EVERY, AttLocation.MIXED_WITH_OPERATION, AttLocation.NO_ATTENTION, AttLocation.DOUBLE_MIXED]:
            return h

        else:
            raise Exception("no match setting")


class Network(nn.Module):
    """
    stack number:layer of cells and then flatten to fed a linear layer
    """

    def __init__(self, c: int, num_classes: int, layers: int, criterion: Module, setting: AttLocation, steps=4,
                 multiplier=4, stem_multiplier=3):
        """
        :param c: 16
        :param num_classes: 10
        :param layers: number of cells of current network
        :param criterion:
        :param steps: nodes num inside cell
        :param multiplier: output channel of cell = multiplier * ch
        :param stem_multiplier: output channel of stem net = stem_multiplier * ch
        """
        super(Network, self).__init__()

        self.c = c
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier
        self.setting = setting

        # stem_multiplier is for stem network,
        # and multiplier is for general cell
        c_curr = stem_multiplier * c  # 3*16
        height_curr = 32  # height of CIFAR
        # stem network, convert 3 channel to c_curr
        self.stem = nn.Sequential(  # 3 => 48
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr)
        )

        # c_curr means a factor of the output channels of current cell
        # output channels = multiplier * c_curr
        cpp, cp, c_curr = c_curr, c_curr, c  # 48, 48, 16
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):

            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                height_curr //= 2
                reduction = True
            else:
                reduction = False

            # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
            # the output channels = multiplier * c_curr
            cell = Cell(steps, multiplier, cpp, cp, c_curr, reduction, reduction_prev, height_curr, height_curr,
                        setting)
            # update reduction_prev
            reduction_prev = reduction

            self.cells += [cell]

            cpp, cp = cp, multiplier * c_curr

        # adaptive pooling output size to 1x1
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.classifier = nn.Linear(cp, num_classes)

        # k is the total number of edges inside single cell, 14
        k = sum(1 for i in range(self.steps) for _ in range(2 + i))

        # define operation and attention parameters
        if self.setting in [AttLocation.AFTER_EVERY, AttLocation.END, AttLocation.NO_ATTENTION, AttLocation.AFTER_EVERY_AND_END]:
            num_first_layers = len(PRIMITIVES)  # 8

        elif self.setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.DOUBLE_MIXED]:
            num_first_layers = len(PRIMITIVES) + len(ATTN_PRIMIVIVES[1:])

        else:
            raise Exception('no match setting')

        if self.setting in [AttLocation.AFTER_EVERY, AttLocation.AFTER_EVERY_AND_END]:
            num_second_layers = len(ATTN_PRIMIVIVES)

        elif self.setting is AttLocation.DOUBLE_MIXED:
            num_second_layers = len(PRIMITIVES) + len(ATTN_PRIMIVIVES[1:])

        elif self.setting in [AttLocation.NO_ATTENTION, AttLocation.END, AttLocation.MIXED_WITH_OPERATION]:
            num_second_layers = 0


        else:
            raise Exception('no match setting')

        # TODO
        # this kind of implementation will add alpha into self.parameters()
        # it has num k of alpha parameters, and each alpha shape: [num_ops]
        # it requires grad and can be converted to cpu/gpu automatically
        # self.alphas_normal = nn.Parameter(torch.randn(k, num_ops))
        # self.alphas_reduce = nn.Parameter(torch.randn(k, num_ops))
        # with torch.no_grad():
        #     # initialize to smaller value
        #     self.alphas_normal.mul_(1e-3)
        #     self.alphas_reduce.mul_(1e-3)

        self.alphas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_first_layers)))
        self.alphas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_first_layers)))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

        if num_second_layers > 0:
            self.betas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_second_layers)))
            self.betas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(k, num_second_layers)))
            self._attn_parameters = [
                self.betas_normal,
                self.betas_reduce
            ]
        else:
            self._attn_parameters = []
            self.betas_normal = self.betas_reduce = None

        if self.setting in [AttLocation.END, AttLocation.AFTER_EVERY_AND_END]:
            self.gammas_normal = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(len(ATTN_PRIMIVIVES))))
            self.gammas_reduce = nn.Parameter(torch.FloatTensor(1e-3 * np.random.randn(len(ATTN_PRIMIVIVES))))
            self._bottleneck_parameters = [
                self.gammas_normal,
                self.gammas_reduce
            ]
        else:
            self._bottleneck_parameters = []
            self.gammas_normal = self.gammas_reduce = None

    def new(self):
        """
        create a new model and initialize it with current alpha parameters.
        However, its weights are left untouched.
        :return:
        """
        model_new = Network(self.c, self.num_classes, self.layers, self.criterion, self.setting).to(device)
        for x, y in zip(model_new.arch_and_attn_parameters(), self.arch_and_attn_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x):
        """
        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        :param x:
        :return:
        """
        # print('in:', x.shape)
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
        # print('stem:', s0.shape)

        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            # according to current cell's type, it choose which architecture parameters
            # to use
            if cell.reduction:  # if current cell is reduction cell
                weights_a = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights_a = F.softmax(self.alphas_normal, dim=-1)  # [14, 8]
            weights_b = None
            if self.betas_normal is not None and self.betas_reduce is not None:
                weights_b = F.softmax(self.betas_reduce, dim=-1) if cell.reduction else F.softmax(self.betas_normal,
                                                                                                  dim=-1)
            weights_c = None
            if self.gammas_normal is not None and self.gammas_reduce is not None:
                weights_c = F.softmax(self.gammas_reduce, dim=-1) if cell.reduction else F.softmax(self.gammas_normal,
                                                                                                   dim=-1)

            # execute cell() firstly and then assign s0=s1, s1=result
            s0, s1 = s1, cell(s0, s1, weights_a, weights_b, weights_c)  # [40, 64, 32, 32]
            # print('cell:',i, s1.shape, cell.reduction, cell.reduction_prev)
            # print('\n')

        # s1 is the last cell's output
        out = self.global_pooling(s1)
        # print('pool', out.shape)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def loss(self, x, target):
        """
        :param x:
        :param target:
        :return:
        """
        logits = self(x)
        return self.criterion(logits, target)

    def arch_and_attn_parameters(self):
        return self._arch_parameters + self._attn_parameters + self._bottleneck_parameters

    def genotype(self) -> Genotype:
        """
        :return:
        """

        def _parse(weights_a, weights_b, weights_c) -> Tuple[list, str]:
            """
            :param weights_a: [14, 8]
            :return:
            """
            gene = []
            n = 2
            start = 0

            # define first and second_layer_primitives
            first_layer_primitives = second_layer_primitives = None
            if self.setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.DOUBLE_MIXED]:
                first_layer_primitives = PRIMITIVES + ATTN_PRIMIVIVES[1:]

            elif self.setting in [AttLocation.NO_ATTENTION, AttLocation.END, AttLocation.AFTER_EVERY,
                                  AttLocation.AFTER_EVERY_AND_END]:
                first_layer_primitives = PRIMITIVES

            else:
                raise Exception("no match setting")

            if self.setting in [AttLocation.AFTER_EVERY, AttLocation.AFTER_EVERY_AND_END]:
                second_layer_primitives = ATTN_PRIMIVIVES

            elif self.setting is AttLocation.DOUBLE_MIXED:
                second_layer_primitives = PRIMITIVES + ATTN_PRIMIVIVES[1:]

            elif self.setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.NO_ATTENTION, AttLocation.END]:
                pass

            else:
                raise Exception("no match setting")

            first_none_index = first_layer_primitives.index('none') if 'none' in first_layer_primitives else -1
            second_none_index = second_layer_primitives.index('none') if second_layer_primitives is not None and 'none' in second_layer_primitives else -1

            for i in range(self.steps):  # for each node
                end = start + n
                W_a: ndarray = weights_a[start:end].copy()  # [2, 8], [3, 8], ... size(k, ops)
                W_b: ndarray = weights_b[
                               start:end].copy() if weights_b is not None else None  # size(k, attns(attn_weights))

                edges = sorted(range(i + 2),  # i+2 is the number of connection for node i
                               key=lambda x: -max(W_a[x][first_layers_idx]  # by descending order
                                                  for first_layers_idx in range(len(first_layer_primitives))
                                                  # get strongest ops
                                                  if first_layers_idx != first_none_index)
                               )[:2]  # only has two inputs(decide 2 edges)
                for j in edges:  # for every input nodes j of current node i

                    best_indices = {'first_layer': None, 'second_layer': None}

                    for k in range(len(first_layer_primitives)):  # get strongest ops for current input j->i
                        if k == first_none_index:
                            continue

                        if best_indices['first_layer'] is None or W_a[j][k] > W_a[j][best_indices['first_layer']]:
                            best_indices['first_layer'] = k

                    length = len(W_b[j]) if W_b is not None else 0
                    for l in range(length):
                        if l == second_none_index:
                            continue

                        if best_indices['second_layer'] is None or W_b[j][l] > W_b[j][best_indices['second_layer']]:
                            best_indices['second_layer'] = l

                    first_layer = first_layer_primitives[
                        best_indices['first_layer']] if first_layer_primitives is not None else ''
                    second_layer = second_layer_primitives[
                        best_indices['second_layer']] if second_layer_primitives is not None else ''

                    gene.append((first_layer, j, second_layer))
                start = end
                n += 1

            bottleneck = ''
            if self.setting in [AttLocation.END, AttLocation.AFTER_EVERY_AND_END]:
                W_c: ndarray = weights_c.copy()
                k_best = None
                for k in range(len(W_c)):  # get strongest ops for current input j->i
                    if k_best is None or W_c[k] > W_c[k_best]:
                        k_best = k
                bottleneck = ATTN_PRIMIVIVES[k_best]
            return gene, bottleneck

        if self.setting in [AttLocation.AFTER_EVERY, AttLocation.DOUBLE_MIXED]:

            gene_normal, normal_bottleneck = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
                F.softmax(self.betas_normal, dim=-1).data.cpu().numpy(),
                None)
            gene_reduce, reduce_bottleneck = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
                F.softmax(self.betas_reduce, dim=-1).data.cpu().numpy(),
                None)

        elif self.setting in [AttLocation.MIXED_WITH_OPERATION, AttLocation.NO_ATTENTION]:
            gene_normal, normal_bottleneck = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
                None,
                None)
            gene_reduce, reduce_bottleneck = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
                None,
                None)

        elif self.setting is AttLocation.END:

            gene_normal, normal_bottleneck = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
                None,
                F.softmax(self.gammas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, reduce_bottleneck = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
                None,
                F.softmax(self.gammas_reduce, dim=-1).data.cpu().numpy())

        elif self.setting is AttLocation.AFTER_EVERY_AND_END:

            gene_normal, normal_bottleneck = _parse(
                F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(),
                F.softmax(self.betas_normal, dim=-1).data.cpu().numpy(),
                F.softmax(self.gammas_normal, dim=-1).data.cpu().numpy())
            gene_reduce, reduce_bottleneck = _parse(
                F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(),
                F.softmax(self.betas_reduce, dim=-1).data.cpu().numpy(),
                F.softmax(self.gammas_reduce, dim=-1).data.cpu().numpy())


        else:
            raise Exception("no match setting")

        concat = range(2 + self.steps - self.multiplier, self.steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat,
            normal_bottleneck=normal_bottleneck, reduce_bottleneck=reduce_bottleneck
        )

        return genotype

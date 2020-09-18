import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from operations import Identity

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

ATTNS = {
    'Identity': lambda c, height, width: Identity(),
    'SE': lambda c, height, width: SqueezeAndExcitation(c),
    'BAM': lambda c, height, width: BottleneckAttentionModule(c),
    'CBAM': lambda c, height, width: ConvolutionalBAM(c),
    'GE_theta-plus': lambda c, height, width: GEBlock(c, height),
    'DoubleAttention': lambda c, height, width: DoubleAttentionLayer(c)
}


# SE (unofficial) https://github.com/moskomule/senet.pytorch
class SEMask(nn.Module):

    def __init__(self, c, r=16):  # Ablation Study Setting
        """

        :param c: input and output channel
        :param r: reduction ratio
        """

        super(SEMask, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x) -> Tensor:
        batch, channel, _, _ = x.size()
        out = self.squeeze(x).view(batch, channel)
        out = self.excitation(out).view(batch, channel, 1, 1)
        out = out.expand_as(x)

        return out


class SqueezeAndExcitation(nn.Module):

    def __init__(self, c, r=16):
        """

        :param c: input and output channel
        :param r: reduction ratio
        """
        super(SqueezeAndExcitation, self).__init__()
        self._mask = SEMask(c, r)

    def forward(self, x) -> Tensor:
        out = self._mask(x) * x

        return out


# BAM https://github.com/Jongchan/attention-module
class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, c, reduction_ratio=16, num_layers=1, paper=False):

        super(ChannelGate, self).__init__()

        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [c]
        gate_channels += [c // reduction_ratio] * num_layers
        gate_channels += [c]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            if not paper:
                self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
                                       nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))
        if paper:
            self.gate_c.add_module('gate_c_bn_final', nn.BatchNorm1d(gate_channels[-1]))

    def forward(self, x):

        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))

        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):

    def __init__(self, c, reduction_ratio=16, dilation_conv_num=2, dilation_val=4, paper=False):

        super(SpatialGate, self).__init__()

        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0',
                               nn.Conv2d(c, c // reduction_ratio, kernel_size=1))
        if not paper:
            self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(c // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i,
                                   nn.Conv2d(c // reduction_ratio, c // reduction_ratio,
                                             kernel_size=3, padding=dilation_val, dilation=dilation_val))
            if not paper:
                self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(c // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(c // reduction_ratio, 1, kernel_size=1))
        if paper:
            self.gate_s.add_module('gate_s_bn_final', nn.BatchNorm2d(1))

    def forward(self, x):

        return self.gate_s(x).expand_as(x)


class BAMMask(nn.Module):

    def __init__(self, c, paper=False):
        super(BAMMask, self).__init__()
        self.channel_att = ChannelGate(c, paper=paper)
        self.spatial_att = SpatialGate(c, paper=paper)
        self.paper = paper

    def forward(self, x) -> Tensor:
        com = self.channel_att(x) * self.spatial_att(x)

        return torch.sigmoid(com) + torch.ones(x.size()).to(device)


class BottleneckAttentionModule(nn.Module):

    def __init__(self, c, paper=False):
        super(BottleneckAttentionModule, self).__init__()
        self._mask = BAMMask(c, paper=paper)

    def forward(self, x) -> Tensor:
        return x * self._mask(x)


# CBAM https://github.com/Jongchan/attention-module
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()

    return outputs


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):

        super(BasicConv, self).__init__()

        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class CBAMChannelAttention(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None):

        super(CBAMChannelAttention, self).__init__()

        if pool_types is None:
            pool_types = ['avg', 'max']
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return scale


class CBAMSpatialAttention(nn.Module):

    def __init__(self):
        super(CBAMSpatialAttention, self).__init__()

        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting

        return scale


class CBAMMask(nn.Module):

    def __init__(self, c, reduction_ratio=16, pool_types=None, no_spatial=False):

        super(CBAMMask, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelAttention = CBAMChannelAttention(c, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialAttention = CBAMSpatialAttention()

    def forward(self, x) -> Tensor:

        c_weight = self.ChannelAttention(x)

        return self.SpatialAttention(c_weight * x) * c_weight


class ConvolutionalBAM(nn.Module):

    def __init__(self, c, reduction_ratio=16, pool_types=None, no_spatial=False):
        super(ConvolutionalBAM, self).__init__()

        self._mask = CBAMMask(c, reduction_ratio, pool_types, no_spatial)

    def forward(self, x):
        return x * self._mask(x)


# GE (unofficial) https://github.com/BayesWatch/pytorch-GENet/blob/master/models/blocks.py
class Downblock(nn.Module):

    def __init__(self, channels, kernel_size=3, relu=True, stride=2, padding=1):
        super(Downblock, self).__init__()

        self.dwconv = nn.Conv2d(channels, channels, groups=channels, stride=stride,
                                kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = relu

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x)

        return x


class GEMask(nn.Module):

    def __init__(self, out_planes, spatial, extent=0, extra_params=True, mlp=True):

        super(GEMask, self).__init__()

        self.extent = extent

        if extra_params is True:
            if extent == 0:
                # Global DW Conv + BN
                self.downop = Downblock(out_planes, relu=False, kernel_size=spatial, stride=1, padding=0)
            elif extent == 2:
                self.downop = Downblock(out_planes, relu=False)

            elif extent == 4:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))
            elif extent == 8:
                self.downop = nn.Sequential(Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=True),
                                            Downblock(out_planes, relu=False))
            else:
                raise NotImplementedError('Extent must be 0,2,4 or 8 for now')
        else:
            if extent == 0:
                self.downop = nn.AdaptiveAvgPool2d(1)
            else:
                self.downop = nn.AdaptiveAvgPool2d(spatial // extent)

        if mlp is True:
            self.mlp = nn.Sequential(nn.Conv2d(out_planes, out_planes // 16, kernel_size=1, padding=0, bias=False),
                                     nn.ReLU(),
                                     nn.Conv2d(out_planes // 16, out_planes, kernel_size=1, padding=0, bias=False),
                                     )
        else:
            self.mlp = Identity()

    def forward(self, x: Tensor) -> Tensor:

        # Assuming squares because lazy.
        shape_in = x.shape[-1]

        # Down, up, sigmoid
        feature_map = self.downop(x)
        feature_map = self.mlp(feature_map)
        feature_map = F.interpolate(feature_map, shape_in)

        return torch.sigmoid(feature_map)


class GEBlock(nn.Module):

    def __init__(self, out_planes, spatial, extent=0, extra_params=True, mlp=True):
        # If extent is zero, assuming global.

        super(GEBlock, self).__init__()
        self._mask = GEMask(out_planes, spatial, extent, extra_params, mlp)

    def forward(self, x):
        out = x * self._mask(x)

        return out


# DoubleAttention of A^2-net (unofficial) https://github.com/gjylt/DoubleAttentionNet/blob/master/DoubleAttentionLayer.py
class DoubleAttentionMask(nn.Module):

    def __init__(self, in_channels, c_m=None, c_n=None, k=1):
        super(DoubleAttentionMask, self).__init__()

        self.K = k
        self.c_m = c_m if c_m is not None else in_channels // 4  # kinetic experiment setting
        self.c_n = c_n if c_n is not None else in_channels // 4
        self.softmax = nn.Softmax(-1)
        self.in_channels = in_channels

        self.convA = nn.Conv2d(in_channels, self.c_m, 1)
        self.convB = nn.Conv2d(in_channels, self.c_n, 1)
        self.convV = nn.Conv2d(in_channels, self.c_n, 1)
        self.convZ = nn.Conv2d(self.c_m, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()

        assert c == self.in_channels, 'input channel not equal!'
        # assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b / self.K)

        tmpA = A.view(batch, self.K, self.c_m, h * w).permute(0, 2, 1, 3).view(batch, self.c_m, self.K * h * w)
        tmpB = B.view(batch, self.K, self.c_n, h * w).permute(0, 2, 1, 3).view(batch * self.c_n,
                                                                               self.K * h * w)
        tmpV = V.view(batch, self.K, self.c_n, h * w).permute(0, 1, 3, 2).contiguous().view(int(b * h * w), self.c_n)

        softmaxB = self.softmax(tmpB).view(batch, self.c_n, self.K * h * w).permute(0, 2, 1)
        softmaxV = self.softmax(tmpV).view(batch, self.K * h * w, self.c_n).permute(0, 2, 1)

        tmpG = tmpA.matmul(softmaxB)
        tmpZ = tmpG.matmul(softmaxV)
        tmpZ = tmpZ.view(batch, self.c_m, self.K, h * w).permute(0, 2, 1, 3).view(int(b), self.c_m, h, w)

        out = self.convZ(tmpZ)

        return out


class DoubleAttentionLayer(nn.Module):

    def __init__(self, in_channels, c_m=None, c_n=None, k=1):
        super(DoubleAttentionLayer, self).__init__()

        self._mask = DoubleAttentionMask(in_channels, c_m, c_n, k)

    def forward(self, x: Tensor) -> Tensor:
        return self._mask(x) + x

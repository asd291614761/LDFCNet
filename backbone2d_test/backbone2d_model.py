import torch
import numpy as np
import torch.autograd
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import DropPath
from timm.models.layers import SqueezeExcite
from easydict import EasyDict
from timm.models.vision_transformer import trunc_normal_


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(
                1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class spatial(torch.nn.Module):
    def __init__(self, ed, k) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, k, 1, pad=k // 2, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.conv3 = Conv2d_BN(ed, ed, 3, 1, pad=1, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x) + self.conv3(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv3 = self.conv3.fuse()

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv3_w = conv3.weight
        conv3_b = conv3.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [2, 2, 2, 2])

        conv3_w = torch.nn.functional.pad(conv3_w, [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + conv3_w + identity
        final_conv_b = conv_b + conv1_b + conv3_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class FC_Block(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size):
        super(FC_Block, self).__init__()

        self.spatial = nn.Sequential(
            spatial(inp, kernel_size),
        )
        self.channel = Residual(
            nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            )
        )
        self.drop = DropPath(0.1)

    def forward(self, x):
        out = self.channel(self.spatial(x))
        out = x + self.drop(out)
        return out


class downsample(nn.Module):
    def __init__(self):
        super(downsample, self).__init__()
        channel = 128
        kernel3 = 3
        kernel1 = 1
        self.down3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=kernel3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel1, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
        )
        self.MP = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.MP1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=1)
        )
        self.relu = nn.GELU()

    def forward(self, x, num):
        if num == 3:
            x_down3 = self.down3(x)
            x_down1 = self.down1(x)
            x_map = self.MP1(x)
            x = x_down3 + x_down1
            x = self.relu(x)
            x = x + x_map
        else:
            x_down3 = self.down3(x)
            x_down1 = self.down1(x)
            x_map = self.MP(x)
            x = x_down3 + x_down1
            x = self.relu(x)
            x = x + x_map

        return x


class downsample2(nn.Module):
    def __init__(self):
        super(downsample2, self).__init__()
        channel = 256
        kernel3 = 3
        kernel1 = 1
        self.down3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
            nn.GELU(),
            nn.Conv2d(channel, channel, kernel_size=kernel3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=kernel1, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(channel, eps=1e-3, momentum=0.01),
        )
        self.MP = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.MP1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=1)
        )
        self.relu = nn.GELU()

    def forward(self, x, num):
        if num == 3:
            x_down3 = self.down3(x)
            x_down1 = self.down1(x)
            x_map = self.MP1(x)
            x = x_down3 + x_down1
            x = self.relu(x)
            x = x + x_map
        else:
            x_down3 = self.down3(x)
            x_down1 = self.down1(x)
            x_map = self.MP(x)
            x = x_down3 + x_down1
            x = self.relu(x)
            x = x + x_map

        return x


class LDFCNet_waymo(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(LDFCNet_waymo, self).__init__()

        kernel3 = 3
        self.num_bev_features = 128

        self.c_block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.relu = nn.GELU()

        self.downsample1 = downsample()
        self.block1 = FC_Block(128, 256, 128, kernel_size=5)
        self.block2 = FC_Block(128, 256, 128, kernel_size=5)
        self.downsample2 = downsample()
        self.block3 = FC_Block(128, 256, 128, kernel_size=5)
        self.block4 = FC_Block(128, 256, 128, kernel_size=5)
        self.block5 = FC_Block(128, 256, 128, kernel_size=5)
        self.block6 = FC_Block(128, 256, 128, kernel_size=5)
        self.block7 = FC_Block(128, 256, 128, kernel_size=5)
        self.block8 = FC_Block(128, 256, 128, kernel_size=5)
        self.downsample3 = downsample()
        self.block9 = FC_Block(128, 256, 128, kernel_size=5)
        self.block10 = FC_Block(128, 256, 128, kernel_size=5)

        self.debloc1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )
        self.debloc2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )
        self.debloc3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = spatial_features

        x_c3 = self.c_block(x)
        x_c1 = self.conv2d_1(x)
        x = x_c3 + x_c1
        x = self.relu(x)

        x_block1_down = self.downsample1(x, 1)
        x_block1_enc = self.block1(x_block1_down)
        x_block1_out = self.block2(x_block1_enc)

        x_block2_down = self.downsample2(x_block1_out, 2)
        x_temp = self.block3(x_block2_down)
        x_temp = self.block4(x_temp)
        x_temp = self.block5(x_temp)
        x_temp = self.block6(x_temp)
        x_temp = self.block7(x_temp)
        x_block2_out = self.block8(x_temp)

        x_block3_down = self.downsample3(x_block2_out, 3)
        x_block3_enc = self.block9(x_block3_down)
        x_block3_out = self.block10(x_block3_enc)

        x_up3 = self.debloc3(x_block3_out)
        x_up3 = x_up3[:, :, :59, :59]
        x_up3 = x_up3 + x_block2_out
        x_up3 = self.relu(x_up3)

        x_up2 = self.debloc2(x_up3)
        x_up2 = x_up2 + x_block1_out
        x_up2 = self.relu(x_up2)

        x_up1 = self.debloc1(x_up2)
        x_up1 = x_up1 + x
        x_up1 = self.relu(x_up1)

        data_dict['spatial_features_2d'] = x_up1

        return data_dict


class LDFCNet_nuscenes(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(LDFCNet_nuscenes, self).__init__()

        kernel3 = 3
        self.num_bev_features = 128

        self.c_block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.relu = nn.GELU()

        self.downsample1 = downsample()
        self.block1 = FC_Block(128, 256, 128, kernel_size=5)
        self.block2 = FC_Block(128, 256, 128, kernel_size=5)
        self.downsample2 = downsample()
        self.block3 = FC_Block(128, 256, 128, kernel_size=5)
        self.block4 = FC_Block(128, 256, 128, kernel_size=5)
        self.block5 = FC_Block(128, 256, 128, kernel_size=5)
        self.block6 = FC_Block(128, 256, 128, kernel_size=5)
        self.block7 = FC_Block(128, 256, 128, kernel_size=5)
        self.block8 = FC_Block(128, 256, 128, kernel_size=5)
        self.downsample3 = downsample()
        self.block9 = FC_Block(128, 256, 128, kernel_size=5)
        self.block10 = FC_Block(128, 256, 128, kernel_size=5)

        self.debloc1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )
        self.debloc2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )
        self.debloc3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
        )

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = spatial_features

        x_c3 = self.c_block(x)
        x_c1 = self.conv2d_1(x)
        x = x_c3 + x_c1
        x = self.relu(x)

        x_block1_down = self.downsample1(x, 1)
        x_block1_enc = self.block1(x_block1_down)
        x_block1_out = self.block2(x_block1_enc)

        x_block2_down = self.downsample2(x_block1_out, 2)
        x_temp = self.block3(x_block2_down)
        x_temp = self.block4(x_temp)
        x_temp = self.block5(x_temp)
        x_temp = self.block6(x_temp)
        x_temp = self.block7(x_temp)
        x_block2_out = self.block8(x_temp)

        x_block3_down = self.downsample3(x_block2_out, 3)
        x_block3_enc = self.block9(x_block3_down)
        x_block3_out = self.block10(x_block3_enc)

        x_up3 = self.debloc3(x_block3_out)
        x_up3 = x_up3[:, :, :45, :45]
        x_up3 = x_up3 + x_block2_out
        x_up3 = self.relu(x_up3)

        x_up2 = self.debloc2(x_up3)
        x_up2 = x_up2 + x_block1_out
        x_up2 = self.relu(x_up2)

        x_up1 = self.debloc1(x_up2)
        x_up1 = x_up1 + x
        x_up1 = self.relu(x_up1)

        data_dict['spatial_features_2d'] = x_up1

        return data_dict


class LDFCNet_puls_nuscense(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(LDFCNet_puls_nuscense, self).__init__()

        kernel3 = 3
        self.num_bev_features = 256

        self.c_block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 2, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels // 2, eps=1e-3, momentum=0.01),
        )
        self.relu = nn.GELU()

        self.downsample1 = downsample2()
        self.block1 = FC_Block(256, 512, 256, kernel_size=5)
        self.block2 = FC_Block(256, 512, 256, kernel_size=5)
        self.downsample2 = downsample2()
        self.block3 = FC_Block(256, 512, 256, kernel_size=5)
        self.block4 = FC_Block(256, 512, 256, kernel_size=5)
        self.block5 = FC_Block(256, 512, 256, kernel_size=5)
        self.block6 = FC_Block(256, 512, 256, kernel_size=5)
        self.block7 = FC_Block(256, 512, 256, kernel_size=5)
        self.block8 = FC_Block(256, 512, 256, kernel_size=5)
        self.downsample3 = downsample2()
        self.block9 = FC_Block(256, 512, 256, kernel_size=5)
        self.block10 = FC_Block(256, 512, 256, kernel_size=5)

        self.debloc1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        )
        self.debloc2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        )
        self.debloc3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        )

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        x = spatial_features

        x_c3 = self.c_block(x)
        x_c1 = self.conv2d_1(x)
        x = x_c3 + x_c1
        x = self.relu(x)

        x_block1_down = self.downsample1(x, 1)
        x_block1_enc = self.block1(x_block1_down)
        x_block1_out = self.block2(x_block1_enc)

        x_block2_down = self.downsample2(x_block1_out, 2)
        x_temp = self.block3(x_block2_down)
        x_temp = self.block4(x_temp)
        x_temp = self.block5(x_temp)
        x_temp = self.block6(x_temp)
        x_temp = self.block7(x_temp)
        x_block2_out = self.block8(x_temp)

        x_block3_down = self.downsample3(x_block2_out, 3)
        x_block3_enc = self.block9(x_block3_down)
        x_block3_out = self.block10(x_block3_enc)

        x_up3 = self.debloc3(x_block3_out)
        x_up3 = x_up3[:, :, :45, :45]
        x_up3 = x_up3 + x_block2_out
        x_up3 = self.relu(x_up3)

        x_up2 = self.debloc2(x_up3)
        x_up2 = x_up2 + x_block1_out
        x_up2 = self.relu(x_up2)

        x_up1 = self.debloc1(x_up2)
        x_up1 = x_up1 + x
        x_up1 = self.relu(x_up1)

        data_dict['spatial_features_2d'] = x_up1

        return data_dict


'''
resbackbone
'''

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out

class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()

#-----------------------------------------
        #测试parameter和Gflops专用
        model_cfg = {
            "NAME": BaseBEVResBackbone,
            "LAYER_NUMS": [ 1, 2, 2 ],
            "LAYER_STRIDES": [ 1, 2, 2 ],
            "NUM_FILTERS": [ 128, 128, 256 ],
            "UPSAMPLE_STRIDES": [ 1, 2, 4 ],
            "NUM_UPSAMPLE_FILTERS": [ 128, 128, 128 ]
        }
        model_cfg = EasyDict(model_cfg)
#--------------------------------------------------
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        self.count = 0

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        self.count +=1
        spatial_features = data_dict['spatial_features']

        # if self.count == 240:
        # import matplotlib.pyplot as plt
        # plt.imshow((torch.sum(spatial_features, dim=1) / spatial_features.shape[1]).squeeze().detach().cpu().numpy())
        # plt.savefig("1.png")

        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            # if self.count == 240:
            # plt.imshow((torch.sum(x, dim=1) / x.shape[1]).squeeze().detach().cpu().numpy())
            # plt.savefig("encord_up{%d}.png" %i)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # if self.count == 240:
        # plt.imshow((torch.sum(ups[0], dim=1) / ups[0].shape[1]).squeeze().detach().cpu().numpy())
        # plt.savefig("up0.png")
        # plt.imshow((torch.sum(ups[1], dim=1) / ups[1].shape[1]).squeeze().detach().cpu().numpy())
        # plt.savefig("up1.png")
        # plt.imshow((torch.sum(ups[2], dim=1) / ups[2].shape[1]).squeeze().detach().cpu().numpy())
        # plt.savefig("up2.png")
        # plt.imshow((torch.sum(x, dim=1) / x.shape[1]).squeeze().detach().cpu().numpy())
        # plt.savefig("finall.png")
        data_dict['spatial_features_2d'] = x

        return data_dict

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
# -----------------------------------------
        # 测试parameter和Gflops专用
        model_cfg = {
            "LAYER_NUMS": [5, 5],
            "LAYER_STRIDES": [1, 2],
            "NUM_FILTERS": [128, 256],
            "UPSAMPLE_STRIDES": [1, 2],
            "NUM_UPSAMPLE_FILTERS": [256, 256]
        }
        model_cfg = EasyDict(model_cfg)
# --------------------------------------------------
        self.model_cfg = model_cfg
        #获取特征提取层的参数
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  #层级卷积数
            layer_strides = self.model_cfg.LAYER_STRIDES #卷积步长
            num_filters = self.model_cfg.NUM_FILTERS #卷积输出通道数
        else:
            layer_nums = layer_strides = num_filters = []
        #获取上采样参数
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS #上采样步长
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES #上采样卷积输出通道数
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()  #存放卷积快列表
        self.deblocks = nn.ModuleList() #存放反卷积快列表
        #创建
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1), #填充0,大小为1
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3,stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()]
            for k in range(layer_nums[idx]): #对于每一个块，重复添加卷积-归一化-激活操作
                cur_layers.extend([nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                                   nn.ReLU()
                                   ])

            self.blocks.append(nn.Sequential(*cur_layers))
            #创建反卷积
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                # 如果步长大于1，或者步长等于1但不希望使用卷积层来进行上采样（即希望使用反卷积层进行上采样）：
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx],upsample_strides[idx],stride=upsample_strides[idx], bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    # 否则，步长等于1且希望使用卷积层进行上采样：
                    stride = np.round(1 / stride).astype(np.int_)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(num_filters[idx], num_upsample_filters[idx],stride,stride=stride, bias=False),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        #如果存在额外的反卷积，则创建
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

'''
Hednet
'''

class DEDBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()

        num_SBB = model_cfg.NUM_SBB
        down_strides = model_cfg.DOWN_STRIDES
        dim = model_cfg.FEATURE_DIM
        assert len(num_SBB) == len(down_strides)

        num_levels = len(down_strides)

        first_block = []
        if input_channels != dim:
            first_block.append(BasicBlock(input_channels, dim, down_strides[0], 1, True))
        first_block += [BasicBlock(dim, dim) for _ in range(num_SBB[0])]
        self.encoder = nn.ModuleList([nn.Sequential(*first_block)])

        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, dim, down_strides[idx], 1, True)]
            cur_layers.extend([BasicBlock(dim, dim) for _ in range(num_SBB[idx])])
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dim, dim, down_strides[idx], down_strides[idx], bias=False),
                    nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                    nn.ReLU()
                )
            )
            self.decoder_norm.append(nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.num_bev_features = dim
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        x = self.encoder[0](x)

        feats = [x]
        for conv in self.encoder[1:]:
            x = conv(x)
            feats.append(x)

        for deconv, norm, up_x in zip(self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)

        data_dict['spatial_features_2d'] = x
        data_dict['spatial_features'] = x
        return data_dict

class CascadeDEDBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()

#-----------------------------------------
        #测试参数量专用
        model_cfg = {
            "NAME": CascadeDEDBackbone,
            "NUM_LAYERS": 4,
            "NUM_SBB": [2, 1, 1],
            "DOWN_STRIDES": [1, 2, 2],
            "FEATURE_DIM": 128
        }
        model_cfg = EasyDict(model_cfg)
#-----------------------------------------

        num_layers = model_cfg.NUM_LAYERS

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            input_dim = input_channels if idx == 0 else model_cfg.FEATURE_DIM
            self.layers.append(DEDBackbone(model_cfg, input_dim))

        self.num_bev_features = model_cfg.FEATURE_DIM

    def forward(self, data_dict):
        for layer in self.layers:
            data_dict = layer(data_dict)
        data_dict['spatial_features_2d'] = data_dict['spatial_features']
        return data_dict
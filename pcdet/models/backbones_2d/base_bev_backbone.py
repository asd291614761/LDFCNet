import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import SqueezeExcite

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
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
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
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

        c_in = sum(num_upsample_filters)
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


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
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

        c_in = sum(num_upsample_filters)
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
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


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
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, device=x.device).ge_(self.drop).div(1 - self.drop).detach()
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
        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [2, 2, 2, 2])

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
            nn.Conv2d(input_channels, input_channels//2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_channels//2, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels//4, eps=1e-3, momentum=0.01),
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
            nn.Conv2d(input_channels, input_channels//2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(input_channels//2, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels // 4, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 4, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels//4, eps=1e-3, momentum=0.01),
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
            nn.Conv2d(input_channels, input_channels//2, kernel_size=kernel3, padding=1, bias=False),
            nn.BatchNorm2d(input_channels//2, eps=1e-3, momentum=0.01),
        )
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_channels//2, eps=1e-3, momentum=0.01),
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
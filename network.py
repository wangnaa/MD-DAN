import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class se_block_conv(nn.Module):
    def __init__(self, channel, kernel, stride, padding, enable):
        super(se_block_conv, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.enable = enable

        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv2_norm = nn.InstanceNorm2d(channel)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        output += x
        output = F.relu(output)
        return output


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        try:
            m.bias.data.zero_()
        except:
            return


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class updata_x2(nn.Module):
    def __init__(self, A):
        super(updata_x2, self).__init__()
        self.A = A

    def forward(self, T2_Gen, underT2_K):
        x2_img = torch.zeros_like(T2_Gen)
        x2 = torch.stack((T2_Gen, x2_img), 4)
        A_ = torch.zeros_like(x2)
        d, m, n, p, q = x2.size()
        for k in range(d):
            A_[k, 0, :, :, 0] = self.A
            A_[k, 0, :, :, 1] = self.A
        x2_k = torch.fft(x2, 2)
        z_k = (1 - A_) * x2_k + underT2_K
        z = torch.ifft(z_k, 2)
        z1, z2 = torch.chunk(z, 2, 4)
        z = torch.sqrt(torch.pow(z1, 2) + torch.pow(z2, 2))
        z = z.view(-1, 1, 240, 240)
        return z


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Block_res(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_res, self).__init__()
        # self.conv1=conv(dim, dim, kernel_size, bias=True)
        # self.act1=nn.ReLU(inplace=True)
        self.res = se_block_conv(dim, 3, 1, 1, False)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.res(x)
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


class Block_res_NoA(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_res_NoA, self).__init__()
        # self.conv1=conv(dim, dim, kernel_size, bias=True)
        # self.act1=nn.ReLU(inplace=True)
        self.res = se_block_conv(dim, 3, 1, 1, False)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

    def forward(self, x):
        res = self.res(x)
        res = self.conv2(res)
        res += x
        return res


class Group_res4_NoA(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Group_res4_NoA, self).__init__()
        self.block = Block_res_NoA(conv, dim, kernel_size)

    def forward(self, x):
        res1 = self.block(x)
        res2 = self.block(res1)
        res3 = self.block(res2)
        res4 = self.block(res3)
        out = torch.cat([res1, res2, res3, res4], dim=1)
        return out


class Group_res5_NoA(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Group_res5_NoA, self).__init__()
        self.block = Block_res_NoA(conv, dim, kernel_size)

    def forward(self, x):
        res1 = self.block(x)
        res2 = self.block(res1)
        res3 = self.block(res2)
        res4 = self.block(res3)
        res5 = self.block(res4)
        out = torch.cat([res1, res2, res3, res4, res5], dim=1)
        return out


class Group_res4(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Group_res4, self).__init__()
        self.block = Block_res(conv, dim, kernel_size)

    def forward(self, x):
        res1 = self.block(x)
        res2 = self.block(res1)
        res3 = self.block(res2)
        res4 = self.block(res3)
        out = torch.cat([res1, res2, res3, res4], dim=1)
        return out


class Group_res5(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Group_res5, self).__init__()
        self.block = Block_res(conv, dim, kernel_size)

    def forward(self, x):
        res1 = self.block(x)
        res2 = self.block(res1)
        res3 = self.block(res2)
        res4 = self.block(res3)
        res5 = self.block(res4)
        out = torch.cat([res1, res2, res3, res4, res5], dim=1)
        return out


class FFA_res(nn.Module):
    def __init__(self, conv=default_conv):
        super(FFA_res, self).__init__()
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(1, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        conv1 = [conv(8 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        self.group4 = Group_res4(conv, self.dim, kernel_size)
        self.group5 = Group_res5(conv, self.dim, kernel_size)
        post_precess = [
            conv(5 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU(),
            conv(self.dim, 1, kernel_size), nn.Sigmoid()]
        self.pre = nn.Sequential(*pre_process)
        self.conv1 = nn.Sequential(*conv1)
        self.post = nn.Sequential(*post_precess)

    def forward(self, T1, T2):
        x1 = self.pre(T1)
        res1 = self.group4(x1)
        x2 = self.pre(T2)
        res2 = self.group4(x2)
        x = torch.cat([res1, res2], dim=1)
        x = self.conv1(x)
        res3 = self.group5(x)
        out = self.post(res3)
        out = out
        return out


class FFA_res_single(nn.Module):
    def __init__(self, conv=default_conv):
        super(FFA_res_single, self).__init__()
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(2, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        conv1 = [conv(4 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        self.group4 = Group_res4(conv, self.dim, kernel_size)
        self.group5 = Group_res5(conv, self.dim, kernel_size)
        post_precess = [
            conv(5 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU(),
            conv(self.dim, 1, kernel_size), nn.Sigmoid()]
        self.pre = nn.Sequential(*pre_process)
        self.conv1 = nn.Sequential(*conv1)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.pre(x)
        res1 = self.group4(x)
        x = self.conv1(res1)
        res2 = self.group5(x)
        out = self.post(res2)
        return out


class FFA_res_NoA(nn.Module):
    def __init__(self, conv=default_conv):
        super(FFA_res_NoA, self).__init__()
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(1, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        conv1 = [conv(8 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        self.group4 = Group_res4_NoA(conv, self.dim, kernel_size)
        self.group5 = Group_res5_NoA(conv, self.dim, kernel_size)
        post_precess = [
            conv(5 * self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU(),
            conv(self.dim, 1, kernel_size), nn.Sigmoid()]
        self.pre = nn.Sequential(*pre_process)
        self.conv1 = nn.Sequential(*conv1)
        self.post = nn.Sequential(*post_precess)

    def forward(self, T1, T2):
        x1 = self.pre(T1)
        res1 = self.group4(x1)
        x2 = self.pre(T2)
        res2 = self.group4(x2)
        x = torch.cat([res1, res2], dim=1)
        x = self.conv1(x)
        res3 = self.group5(x)
        out = self.post(res3)
        out = out
        return out









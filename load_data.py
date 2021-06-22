import torch, math
# import pytorch_fft.fft as fft
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

        # self.se_conv1 = nn.Conv2d(channel, channel // 16, kernel_size=1)
        # self.se_conv2 = nn.Conv2d(channel // 16, channel, kernel_size=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))

        # if self.enable:
        #     se = F.avg_pool2d(output, output.size(2))
        #     se = F.relu(self.se_conv1(se))
        #     se = F.sigmoid(self.se_conv2(se))
        #     output = output * se

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

class Generator(nn.Module):
    def __init__(self,num_blocks,num_chanels):
        super(Generator,self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(1, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2*self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(2*self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2*self.num_chanels, 4*self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(4*self.num_chanels),
                                   nn.ReLU())

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(se_block_conv(4 * self.num_chanels, 3, 1, 1, False))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels,1,3,1,1),nn.Sigmoid())
        
    def forward(self, T1):
        x = self.conv1(T1)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.blocks(x)
        
        x = self.deconv1(x)
        x = self.deconv2(x)
        
        T2_Gen = self.out(x)
        
        return T2_Gen

class Generator_cat(nn.Module):
    def __init__(self, num_blocks, num_chanels):
        super(Generator_cat, self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(2, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(se_block_conv(4 * self.num_chanels, 3, 1, 1, False))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, T1, T2_re):
        T = torch.cat((T1, T2_re), 1)
        x = self.conv1(T)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.blocks(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        T2_Gen = self.out(x)

        return T2_Gen

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

class Block_res_attention(nn.Module):
    def __init__(self, dim):
        super(Block_res_attention, self).__init__()
        self.res = se_block_conv(dim, 3, 1, 1, False)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.res(x)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x
        return res

class res_attention(nn.Module):
    def __init__(self, num_blocks, num_chanels):
        super(res_attention, self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(2, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(Block_res_attention(4 * self.num_chanels))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, T1, T2_re):
        T = torch.cat((T1, T2_re), 1)
        x = self.conv1(T)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.blocks(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        T2_Gen = self.out(x)

        return T2_Gen

class res_attention_2(nn.Module):
    def __init__(self, num_blocks, num_chanels):
        super(res_attention_2, self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(1, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())
        self.conv4 = nn.Conv2d(8 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1)

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(Block_res_attention(4 * self.num_chanels))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, T1, T2):
        x1 = self.conv1(T1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.blocks(x1)

        x2 = self.conv1(T2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.blocks(x2)

        x = torch.cat((x1,x2),1)

        x = self.conv4(x)
        x = self.blocks(x)
        x = self.blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)

        T2_Gen = self.out(x)

        return T2_Gen

class res_attention2(nn.Module):
    def __init__(self, num_blocks, num_chanels):
        super(res_attention2, self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(1, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())
        self.conv4 = nn.Conv2d(8 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1)

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(Block_res_attention(4 * self.num_chanels))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, T1, T2):
        x1 = self.conv1(T1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.blocks(x1)

        x2 = self.conv1(T2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.blocks(x2)

        x = torch.cat((x1,x2),1)

        x = self.conv4(x)
        x = self.blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)

        T2_Gen = self.out(x)

        return T2_Gen

class Re_Pro(nn.Module):
    def __init__(self, A, num_blocks, num_chanels):
        super(Re_Pro, self).__init__()
        self.A = A
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(3, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(se_block_conv(4 * self.num_chanels, 3, 1, 1, False))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, T2_G, under_T2, z):
        top = abs(np.fft.ifft2(np.fft.fft2(under_T2) * self.A.T))
        top = torch.Tensor(top)
        top = top.view(-1, 1, 240, 240)
        top = top.cuda()
        x = torch.cat((T2_G, top, z), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.blocks(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        z = self.out(x)

        return z

class Proximal(nn.Module):
    def __init__(self, num_blocks, num_chanels):
        super(Proximal, self).__init__()
        self.num_blocks = num_blocks
        self.num_chanels = num_chanels

        self.conv1 = nn.Sequential(nn.Conv2d(1, self.num_chanels, 3, 1, 1), nn.InstanceNorm2d(self.num_chanels),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(2 * self.num_chanels),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(2 * self.num_chanels, 4 * self.num_chanels, 3, 1, 1),
                                   nn.InstanceNorm2d(4 * self.num_chanels),
                                   nn.ReLU())

        blocks = []
        for i in range(self.num_blocks):
            blocks.append(se_block_conv(4 * self.num_chanels, 3, 1, 1, False))
        self.blocks = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(4 * self.num_chanels, 2 * self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(2 * self.num_chanels), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2 * self.num_chanels, self.num_chanels, 3, 1, 1),
                                     nn.InstanceNorm2d(self.num_chanels), nn.ReLU())
        self.out = nn.Sequential(nn.Conv2d(self.num_chanels, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, under_T2):
        x = self.conv1(under_T2)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.blocks(x)

        x = self.deconv1(x)
        x = self.deconv2(x)

        T2_Re = self.out(x)

        return T2_Re

class Reconstruction(nn.Module):
    def __init__(self, A, G, P, R):
        super(Reconstruction, self).__init__()
        self.A = A
        self.G = G
        self.P = P
        self.R = R
        self.rho = torch.nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.rho, self.P)
        self.mu = torch.nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.mu, self.G)
        self.nu = torch.nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.nu, self.G)

    def forward(self, T2_Gen, T2_Re, underT2, z):
        top = abs(np.fft.ifft2(np.fft.fft2(underT2) * self.A))
        top = torch.Tensor(top)
        top = top.view(-1, 1, 240, 240)
        top = top.cuda()
        top = top + self.rho * T2_Gen + self.nu * T2_Re + self.mu * z

        bottom = abs(np.fft.ifft2(np.fft.fft2(np.identity(240)) * self.A * self.A.T))
        bottom = torch.Tensor(bottom)
        bottom = bottom.cuda()
        bottom = bottom + self.rho + self.mu + self.nu

        T2_Rec = torch.div(top, bottom)

        return T2_Rec

class updata_z(nn.Module):
    def __init__(self, A):
        super(updata_z, self).__init__()
        self.A = A

    def forward(self, T2_Gen, underT2_K):
        x2 = T2_Gen.detach()
        x2 = x2.cpu()
        x2 = x2.view(-1, 240, 240)
        y_k = underT2_K.numpy()
        # y_k = y_k.view(-1, 2, 240, 240)
        d, m, n = x2.size()
        x2_k = np.zeros((d, m, n), complex)
        z_k = np.zeros((d, m, n), complex)
        z = np.zeros((d, m, n))
        for k in range(d):
            x2_k[k,:,:] = np.fft.fft2(x2[k,:,:])
            z_k[k,:,:] = (1 - self.A) * x2_k[k,:,:] + y_k[k,0,:,:] + y_k[k,1,:,:] * 1j
            # for i in range(m):
            #     for j in range(n):
            #         if self.A[i,j] == 1:
            #             z_k[k,i,j] = y_k[k,i,j]
            #         if self.A[i,j] == 0:
            #             z_k[k,i,j] = x2_k[k,i,j]
            z[k,:,:] = abs(np.fft.ifft2(z_k[k,:,:]))
        z = torch.Tensor(z)
        z = z.view(-1,1,240,240)
        z = z.cuda()
        return z

class updata_x2(nn.Module):
    def __init__(self, A):
        super(updata_x2, self).__init__()
        self.A = A

    def forward(self, T2_Gen, underT2_K):
        x2_img = torch.zeros_like(T2_Gen)
        x2 = torch.stack((T2_Gen,x2_img),4)
        A_ = torch.zeros_like(x2)
        d, m, n, p, q = x2.size()
        for k in range(d):
            A_[k, 0, :, :, 0] = self.A
            A_[k, 0, :, :, 1] = self.A
        x2_k = torch.fft(x2,2)
        z_k = (1 - A_) * x2_k + underT2_K
        z = torch.ifft(z_k,2)
        z1, z2 = torch.chunk(z, 2, 4)
        z = torch.sqrt(torch.pow(z1, 2) + torch.pow(z2, 2))
        z = z.view(-1,1,240,240)
        return z

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.nChannels = nChannels
        self.growthRate = growthRate

        self.bn1 = nn.BatchNorm2d(self.nChannels)
        self.conv1 = nn.Conv2d(self.nChannels, self.growthRate, 3, 1, 1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Dense_block(nn.Module):
    def __init__(self, growthRate, depth):
        super(Dense_block, self).__init__()
        self.growthRate = growthRate
        self.depth = depth

        nChannels = self.growthRate
        self.dense = self._make_dense(nChannels, self.growthRate, self.depth)

        nChannels += self.depth * self.growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, depth):
        layers = []
        for i in range(int(depth)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dense(x)
        out = F.relu(self.bn1(out))
        return out

class Dense_Unet(nn.Module):
    def __init__(self, nchannels, depth):
        super(Dense_Unet, self).__init__()
        self.nchannels = nchannels
        self.depth = depth

        self.pre_feature_ex = nn.Sequential(nn.Conv2d(2, 4 * self.nchannels, 3, 1, 1),
                                            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.down_conv = nn.Sequential(
            nn.Conv2d(4 * self.nchannels, self.nchannels, 3, 1, 1),
            nn.BatchNorm2d(self.nchannels), nn.ReLU())

        self.up_conv = nn.Sequential(
            nn.Conv2d(8 * self.nchannels, self.nchannels, 3, 1, 1),
            nn.BatchNorm2d(self.nchannels), nn.ReLU())

        self.dense_block = Dense_block(self.nchannels, self.depth)

        self.down_transition_conv = nn.Sequential(
            nn.Conv2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=1, padding=0),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())
        self.down_transition_pool = nn.AvgPool2d(kernel_size=2, padding=0)

        self.up_transition = nn.Sequential(
            nn.ConvTranspose2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.rec = nn.Sequential(nn.Conv2d(5 * self.nchannels, 1, 3, 1, 1), nn.BatchNorm2d(1),
                                 nn.ReLU())

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x_down1 = self.pre_feature_ex(x)
        x_down1 = self.down_conv(x_down1)
        x_down1 = self.dense_block(x_down1)
        x_down1 = self.down_transition_conv(x_down1)

        x_down2 = self.down_transition_pool(x_down1)
        x_down2 = self.down_conv(x_down2)
        x_down2 = self.dense_block(x_down2)
        x_down2 = self.down_transition_conv(x_down2)

        x_up2 = self.down_transition_pool(x_down2)
        x_up2 = self.down_conv(x_up2)
        x_up2 = self.dense_block(x_up2)
        x_up2 = self.up_transition(x_up2)

        x_up2 = torch.cat((x_down2, x_up2), 1)

        x_up1 = self.up_conv(x_up2)
        x_up1 = self.dense_block(x_up1)
        x_up1 = self.up_transition(x_up1)

        x_up1 = torch.cat((x_down1, x_up1), 1)

        out = self.up_conv(x_up1)
        out = self.dense_block(out)
        out = self.rec(out)

        return out

class Dense_Unet_1(nn.Module):
    def __init__(self, nchannels, depth):
        super(Dense_Unet_1, self).__init__()
        self.nchannels = nchannels
        self.depth = depth

        self.pre_feature_ex = nn.Sequential(nn.Conv2d(1, 4 * self.nchannels, 3, 1, 1),
                                            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.down_conv = nn.Sequential(
            nn.Conv2d(4 * self.nchannels, self.nchannels, 3, 1, 1),
            nn.BatchNorm2d(self.nchannels), nn.ReLU())

        self.up_conv = nn.Sequential(
            nn.Conv2d(8 * self.nchannels, self.nchannels, 3, 1, 1),
            nn.BatchNorm2d(self.nchannels), nn.ReLU())

        self.dense_block = Dense_block(self.nchannels, self.depth)

        self.down_transition_conv = nn.Sequential(
            nn.Conv2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=1, padding=0),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())
        self.down_transition_pool = nn.AvgPool2d(kernel_size=2, padding=0)

        self.up_transition = nn.Sequential(
            nn.ConvTranspose2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.rec = nn.Sequential(nn.Conv2d(5 * self.nchannels, 1, 3, 1, 1), nn.BatchNorm2d(1),
                                 nn.ReLU())

    def forward(self, x2):
        # x = torch.cat((x1, x2), 1)
        x_down1 = self.pre_feature_ex(x2)
        x_down1 = self.down_conv(x_down1)
        x_down1 = self.dense_block(x_down1)
        x_down1 = self.down_transition_conv(x_down1)

        x_down2 = self.down_transition_pool(x_down1)
        x_down2 = self.down_conv(x_down2)
        x_down2 = self.dense_block(x_down2)
        x_down2 = self.down_transition_conv(x_down2)

        x_up2 = self.down_transition_pool(x_down2)
        x_up2 = self.down_conv(x_up2)
        x_up2 = self.dense_block(x_up2)
        x_up2 = self.up_transition(x_up2)

        x_up2 = torch.cat((x_down2, x_up2), 1)

        x_up1 = self.up_conv(x_up2)
        x_up1 = self.dense_block(x_up1)
        x_up1 = self.up_transition(x_up1)

        x_up1 = torch.cat((x_down1, x_up1), 1)

        out = self.up_conv(x_up1)
        out = self.dense_block(out)
        out = self.rec(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, growth_rate, nb_layers):
        super(DenseBlock, self).__init__()
        in_planes = growth_rate
        self.layer = self._make_layer(in_planes, growth_rate, nb_layers)
        self.bn1 = nn.BatchNorm2d(in_planes+nb_layers*growth_rate)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def _make_layer(self, in_planes, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(in_planes+i*growth_rate, growth_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layer(x)
        out = self.relu(self.bn1(out))
        return out

class DenseUnet(nn.Module):
    def __init__(self, nchannels, depth):
        super(DenseUnet, self).__init__()
        self.nchannels = nchannels
        self.depth = depth

        self.pre_feature_ex = nn.Sequential(nn.Conv2d(6, 4 * self.nchannels, 3, 1, 1),
                                            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.down_conv = nn.Sequential(nn.Conv2d(4 * self.nchannels, self.nchannels, 3, 1, 1))

        self.up_conv = nn.Sequential(
            nn.Conv2d(8 * self.nchannels, self.nchannels, 3, 1, 1),
            nn.BatchNorm2d(self.nchannels), nn.ReLU())

        self.denseBlock = DenseBlock(self.nchannels, self.depth)

        self.down_transition_conv = nn.Sequential(
            nn.Conv2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=1, padding=0),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())
        self.down_transition_pool = nn.AvgPool2d(kernel_size=2, padding=0)

        self.up_transition = nn.Sequential(
            nn.ConvTranspose2d(5 * self.nchannels, 4 * self.nchannels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.nchannels), nn.ReLU())

        self.rec = nn.Sequential(nn.Conv2d(5 * self.nchannels, 3, 3, 1, 1), nn.BatchNorm2d(3))

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x_down1 = self.pre_feature_ex(x)
        x_down1 = self.down_conv(x_down1)
        x_down1 = self.denseBlock(x_down1)
        x_down1 = self.down_transition_conv(x_down1)

        x_down2 = self.down_transition_pool(x_down1)
        x_down2 = self.down_conv(x_down2)
        x_down2 = self.denseBlock(x_down2)
        x_down2 = self.down_transition_conv(x_down2)

        x_up2 = self.down_transition_pool(x_down2)
        x_up2 = self.down_conv(x_up2)
        x_up2 = self.denseBlock(x_up2)
        x_up2 = self.up_transition(x_up2)

        x_up2 = torch.cat((x_down2, x_up2), 1)

        x_up1 = self.up_conv(x_up2)
        x_up1 = self.denseBlock(x_up1)
        x_up1 = self.up_transition(x_up1)

        x_up1 = torch.cat((x_down1, x_up1), 1)

        out = self.up_conv(x_up1)
        out = self.denseBlock(out)
        out = self.rec(out)

        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x
        return res

class Block_res(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block_res, self).__init__()
        # self.conv1=conv(dim, dim, kernel_size, bias=True)
        # self.act1=nn.ReLU(inplace=True)
        self.res = se_block_conv(dim, 3, 1, 1, False)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.res(x)
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x
        return res

class Block_res_NoA(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block_res_NoA, self).__init__()
        # self.conv1=conv(dim, dim, kernel_size, bias=True)
        # self.act1=nn.ReLU(inplace=True)
        self.res = se_block_conv(dim, 3, 1, 1, False)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
    def forward(self, x):
        res=self.res(x)
        res=self.conv2(res)
        res += x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class Group_res(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Group_res, self).__init__()
        self.block = Block_res(conv, dim, kernel_size)
    def forward(self, x):
        res1 = self.block(x)
        res2 = self.block(res1)
        res3 = self.block(res2)
        res4 = self.block(res3)
        out = torch.cat([res1, res2, res3, res4], dim=1)
        return out

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
        self.dim=64
        kernel_size=3
        pre_process = [conv(2, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        conv1 = [conv(4*self.dim, self.dim ,kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU()]
        self.group4 = Group_res4(conv, self.dim, kernel_size)
        self.group5 = Group_res5(conv, self.dim, kernel_size)
        post_precess = [
            conv(5*self.dim, self.dim, kernel_size), nn.InstanceNorm2d(self.dim), nn.ReLU(),
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
        out=self.post(res2)
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









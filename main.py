import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import load_data
import network
import os
import time
from tensorboardX import SummaryWriter
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

save_path = 'snapshot'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# log_dir_path = 'log'
# if not os.path.exists(log_dir_path):
#     os.mkdir(log_dir_path)

# W = SummaryWriter(log_dir=log_dir_path, comment='log_dir')

######### hyper-parameters
batch_size = 2
learning_rate = 1e-4
weight_decay = 1e-5
num_epoches = 61
test_interval = 10

num_chanel_Res = 16
num_blocks_Res = 9
num_blocks_Res2 = 6

num_chanel_Dense = 16
depth_Dense = 4

gps=3
blocks=19

Mask = sio.loadmat('/home/mydell/WN/net/T1_to_T2/Mask_16.mat')['Mask']
Mask = torch.Tensor(Mask)
P = 1.0
G = 1.0
R = 1.0

train_path = "/home/mydell/WN/data/sliceT1.txt"
test_path = "/home/mydell/WN/data/T1_mat/Test.txt"

######### data_loader and network
train_loader = torch.utils.data.DataLoader(load_data.Train_List(train_path),batch_size=2,shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(load_data.Test_List(test_path),batch_size=1,shuffle=True, num_workers=1,drop_last=True)
# modelG_0 = network.Generator_cat(num_blocks_Res,num_chanel_Res).cuda()
# modelG_0 = torch.load(save_path+'/modelG_0.pk')
# modelG_1 = torch.load(save_path+'/modelG_1.pk')
# modelG_2 = torch.load(save_path+'/modelG_2.pk')
# modelG_3 = torch.load(save_path+'/modelG_3.pk')
# modelG_4 = torch.load(save_path+'/modelG_4.pk')
# modelG_5 = torch.load(save_path+'/modelG_5.pk')
# modelG_6 = torch.load(save_path+'/modelG_6.pk')
# modelG_7 = torch.load(save_path+'/modelG_7.pk')
# modelG_8 = torch.load(save_path+'/modelG_7.pk')

# modelGA_0 = network.res_attention(num_blocks_Res,num_chanel_Res).cuda()
# modelGA_0 = torch.load(save_path+'/modelGA_0.pk')
# modelGA2_0 = network.res_attention2(num_blocks_Res2,num_chanel_Res).cuda()
# modelGA2_0 = torch.load(save_path+'/modelGA2_0_89.pk')

# modelG_16_0 = network.Generator_cat(num_blocks_Res,num_chanel_Res).cuda()
# modelG_16_0 = torch.load(save_path+'/modelG_16_0.pk')
# modelG_16_1 = torch.load(save_path+'/modelG_16_1.pk')
# modelG_16_2 = torch.load(save_path+'/modelG_16_2.pk')
# modelG_16_3 = torch.load(save_path+'/modelG_16_3.pk')
# modelG_16_4 = torch.load(save_path+'/modelG_16_4.pk')
# modelG_16_5 = torch.load(save_path+'/modelG_16_5.pk')
# modelG_16_6 = torch.load(save_path+'/modelG_16_6.pk')
# modelG_16_7 = torch.load(save_path+'/modelG_16_7.pk')
# modelG_16_8 = torch.load(save_path+'/modelG_16_8.pk')


# modelDU_0 = network.Dense_Unet(num_chanel_Dense, depth_Dense).cuda()
# modelDU_0 = torch.load(save_path+'/modelDU_0.pk')
# modelDU_1 = torch.load(save_path+'/modelDU_1.pk')
# modelDU_2 = torch.load(save_path+'/modelDU_2.pk')
# modelDU_3 = torch.load(save_path+'/modelDU_3.pk')
# modelDU_4 = torch.load(save_path+'/modelDU_4.pk')
# modelDU_5 = torch.load(save_path+'/modelDU_5.pk')
# modelDU_6 = torch.load(save_path+'/modelDU_6.pk')
# modelDU_7 = torch.load(save_path+'/modelDU_7.pk')
# modelDU_8 = torch.load(save_path+'/modelDU_7.pk')

modelFFA_res_0 = torch.load(save_path+'/modelFFA_res16_0_49.pk')
# modelFFA_res_0 = network.FFA_res().cuda()
# modelFFA_res_0 = torch.load(save_path+'/modelFFA_res_0_50.pk')
# modelFFA_res_1 = torch.load(save_path+'/modelFFA_res_21.pk')
# modelFFA_res_2 = torch.load(save_path+'/modelFFA_res_22.pk')
# modelFFA_res_3 = torch.load(save_path+'/modelFFA_res_22.pk')

# modelFFA_res_0 = torch.load(save_path+'/modelFFA_res_NoA_20.pk')
# modelFFA_res_1 = torch.load(save_path+'/modelFFA_res_NoA_21.pk')
# modelFFA_res_2 = torch.load(save_path+'/modelFFA_res_NoA_22.pk')
# modelFFA_res_3 = torch.load(save_path+'/modelFFA_res_NoA_22.pk')

# modelFFA_res_single_0 = network.FFA_res_single().cuda()

# modelFFA_res_NoA_0 = network.FFA_res_NoA().cuda()


# modelupdata_x2 = network.updata_x2(Mask).cuda()

# modelDU = network.DenseUnet(num_chanel_Dense, depth_Dense).cuda()
# modelP = network.Proximal(num_blocks_pro, num_chanel).cuda()
# modelR_P = network.Re_Pro(Mask, num_blocks_gen, num_chanel).cuda()
# modelR11 = network.Reconstruction1(Mask, G, P).cuda()
# modelR11 = network.Reconstruction(Mask, G, P, R).cuda()

######optimizer and loss
# optimizerDU = torch.optim.Adam(modelDU.parameters(),lr=learning_rate,weight_decay=weight_decay)

# optimizerGA2 = torch.optim.Adam(modelGA2_0.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizerG_1 = torch.optim.Adam(modelG_1.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_2 = torch.optim.Adam(modelG_2.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_3 = torch.optim.Adam(modelG_3.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_4 = torch.optim.Adam(modelG_4.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_6 = torch.optim.Adam(modelG_6.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_7 = torch.optim.Adam(modelG_7.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerG_8 = torch.optim.Adam(modelG_8.parameters(),lr=learning_rate,weight_decay=weight_decay)

# optimizerDU_0 = torch.optim.Adam(modelDU_0.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_1 = torch.optim.Adam(modelDU_1.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_2 = torch.optim.Adam(modelDU_2.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_3 = torch.optim.Adam(modelDU_3.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_4 = torch.optim.Adam(modelDU_4.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_5 = torch.optim.Adam(modelDU_5.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_6 = torch.optim.Adam(modelDU_6.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_7 = torch.optim.Adam(modelDU_7.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerDU_8 = torch.optim.Adam(modelDU_8.parameters(),lr=learning_rate,weight_decay=weight_decay)

optimizerFFA_res_0 = torch.optim.Adam(modelFFA_res_0.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_1 = torch.optim.Adam(modelFFA_res_1.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_2 = torch.optim.Adam(modelFFA_res_2.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_3 = torch.optim.Adam(modelFFA_res_3.parameters(),lr=learning_rate,weight_decay=weight_decay)

loss = nn.MSELoss()
# MAE = nn.L1Loss()



# ########  training 1
# z = torch.zeros(1, 240, 240)
# for epoch in range(num_epoches):
#     ####adjust learning_rate
#     if epoch % 200 == 0:
#         for param_group in optimizerP11.param_groups:
#             param_group['lr'] *= 0.1
#         for param_group in optimizerG_c.param_groups:
#             param_group['lr'] *= 0.1
#
#     if epoch % 200 == 0:
#         for param_group in optimizerR11.param_groups:
#             param_group['lr'] *= 0.1
#
#     # if epoch % 300 == 0:
#     #     for param_group in optimizerG.param_groups:
#     #         param_group['lr'] *= 0.1
#
#     for i,(T1,T2,under_T2) in enumerate(data_loader):
#         T1 = T1.view(-1,1,240,240)
#         T1 = T1.cuda()
#         T2 = T2.view(-1,1,240,240)
#         T2 = T2.cuda()
#         under_T2 = under_T2.view(-1, 1, 240, 240)
#         under_T2_c = under_T2.cuda()
#         z_0 = z.view(-1,1,240,240)
#         z_0 = z_0.cuda()
#
#         with torch.no_grad():
#             x1_Re = modelP(under_T2_c)
#             x1_Re = x1_Re.detach()
#         x1_G = modelG_c(T1, x1_Re)
#         # x1_G = x1_G.detach()
#         # x1_1 = modelR11(x1_G, x1_Re, under_T2, z_0)
#         x1_1 = modelR11(x1_G, under_T2, z_0)
#         z_1 = modelP11(x1_1)
#
#
#         optimizerG_c.zero_grad()
#         optimizerR11.zero_grad()
#         optimizerP11.zero_grad()
#         l = loss(z_1, T2)
#         # l = 0.1 * loss(x1_1, T2) + 0.9 * loss(z_1, T2)
#         l.backward()
#         optimizerG_c.step()
#         optimizerR11.step()
#         optimizerP11.step()
#
#         if i % 10 == 0:
#             print(modelR11.rho.data, modelR11.mu.data)
#             # print(modelR11.rho.data, modelR11.nu.data, modelR11.mu.data)
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l.item()))
#
#     if epoch % test_interval == 0:
#         # l_test = 0
#         l_z1 = 0
#         psnr = 0
#         ssim = 0
#         with torch.no_grad():
#             for T1_test,T2_test,under_T2_test in test_loader:
#                 T1_test = T1_test.view(-1,1,240,240)
#                 T1_test = T1_test.cuda()
#                 T2_test = T2_test.view(-1,1,240,240)
#                 T2_test = T2_test.cuda()
#                 under_T2_test = under_T2_test.view(-1, 1, 240, 240)
#                 under_T2_test_c = under_T2_test.cuda()
#                 z_0_test = z.view(-1, 1, 240, 240)
#                 z_0_test = z_0_test.cuda()
#
#                 x1_Re_test = modelP(under_T2_test_c)
#                 x1_G_test = modelG_c(T1_test, x1_Re_test)
#                 x1_1_test = modelR11(x1_G_test, under_T2_test, z_0_test)
#                 # x1_1_test = modelR11(x1_G_test, x1_Re_test, under_T2_test, z_0_test)
#                 z_1_test = modelP11(x1_1_test)
#
#                 l_z1 += loss(z_1_test, T2_test)
#                 # l_test += 0.1 * loss(x1_1_test, T2_test) + 0.9 * loss(z_1_test ,T2_test)
#                 z_1_test_ = z_1_test.cpu()
#                 z_1_test_ = z_1_test_.data.numpy().squeeze()
#                 T2_test_ = T2_test.cpu().numpy().squeeze()
#                 psnr += compare_psnr(T2_test_, z_1_test_, data_range=T2_test_.max())
#                 ssim += compare_ssim(T2_test_, z_1_test_, data_range=T2_test_.max())
#
#             # l_test /= len(test_loader)
#             l_z1 /= len(test_loader)
#             psnr /= len(test_loader)
#             ssim /= len(test_loader)
#
#         print('epoch:{:d},\tz_loss:{:.4f},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(epoch,100*l_z1.item(),psnr.item(),ssim.item()))
#         # torch.save(modelG, save_path + '/modelG1.pk')
#         torch.save(modelP11, save_path + '/modelP11_c_G.pk')
#         torch.save(modelR11, save_path + '/modelR11_c_G.pk')





# ########  training R_P 1
# z = torch.zeros(2, 240, 240)
# z_ = torch.zeros(1, 240, 240)
# for epoch in range(num_epoches):
#     ####adjust learning_rate
#     if epoch % 200 == 0:
#         for param_group in optimizerR_P.param_groups:
#             param_group['lr'] *= 0.1
#         # for param_group in optimizerG_c.param_groups:
#         #     param_group['lr'] *= 0.1
#
#     # if epoch % 200 == 0:
#     #     for param_group in optimizerR11.param_groups:
#     #         param_group['lr'] *= 0.1
#
#     # if epoch % 300 == 0:
#     #     for param_group in optimizerG.param_groups:
#     #         param_group['lr'] *= 0.1
#
#     for i,(T1,T2,under_T2) in enumerate(data_loader):
#         T1 = T1.view(-1,1,240,240)
#         T1 = T1.cuda()
#         T2 = T2.view(-1,1,240,240)
#         T2 = T2.cuda()
#         under_T2 = under_T2.view(-1, 1, 240, 240)
#         under_T2_c = under_T2.cuda()
#         z_0 = z.view(2,1,240,240)
#         z_0 = z_0.cuda()
#
#         with torch.no_grad():
#             x1_Re = modelP(under_T2_c)
#             x1_Re = x1_Re.detach()
#         x1_G = modelG_c(T1, x1_Re)
#         # x1_G = x1_G.detach()
#         # x1_1 = modelR11(x1_G, x1_Re, under_T2, z_0)
#         # x1_1 = modelR11(x1_G, under_T2, z_0)
#         z_1 = modelR_P(x1_G, under_T2, z_0)
#
#
#         optimizerG_c.zero_grad()
#         # optimizerR11.zero_grad()
#         optimizerR_P.zero_grad()
#         l = loss(z_1, T2)
#         # l = 0.1 * loss(x1_1, T2) + 0.9 * loss(z_1, T2)
#         l.backward()
#         optimizerG_c.step()
#         optimizerR_P.step()
#         # optimizerP11.step()
#
#         if i % 10 == 0:
#             # print(modelR11.rho.data, modelR11.mu.data)
#             # print(modelR11.rho.data, modelR11.nu.data, modelR11.mu.data)
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l.item()))
#
#     if epoch % test_interval == 0:
#         # l_test = 0
#         l_z1 = 0
#         psnr = 0
#         ssim = 0
#         with torch.no_grad():
#             for T1_test,T2_test,under_T2_test in test_loader:
#                 T1_test = T1_test.view(-1,1,240,240)
#                 T1_test = T1_test.cuda()
#                 T2_test = T2_test.view(-1,1,240,240)
#                 T2_test = T2_test.cuda()
#                 under_T2_test = under_T2_test.view(-1, 1, 240, 240)
#                 under_T2_test_c = under_T2_test.cuda()
#                 z_0_test = z_.view(-1, 1, 240, 240)
#                 z_0_test = z_0_test.cuda()
#
#                 x1_Re_test = modelP(under_T2_test_c)
#                 x1_G_test = modelG_c(T1_test, x1_Re_test)
#                 # x1_1_test = modelR11(x1_G_test, under_T2_test, z_0_test)
#                 # x1_1_test = modelR11(x1_G_test, x1_Re_test, under_T2_test, z_0_test)
#                 z_1_test = modelR_P(x1_G_test, under_T2_test, z_0_test)
#
#                 l_z1 += loss(z_1_test, T2_test)
#                 # l_test += 0.1 * loss(x1_1_test, T2_test) + 0.9 * loss(z_1_test ,T2_test)
#                 z_1_test_ = z_1_test.cpu()
#                 z_1_test_ = z_1_test_.data.numpy().squeeze()
#                 T2_test_ = T2_test.cpu().numpy().squeeze()
#                 psnr += compare_psnr(T2_test_, z_1_test_, data_range=T2_test_.max())
#                 ssim += compare_ssim(T2_test_, z_1_test_, data_range=T2_test_.max())
#
#             # l_test /= len(test_loader)
#             l_z1 /= len(test_loader)
#             psnr /= len(test_loader)
#             ssim /= len(test_loader)
#
#         print('epoch:{:d},\tz_loss:{:.4f},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(epoch,100*l_z1.item(),psnr.item(),ssim.item()))
#         W.add_scalar('test_loss', 100*l_z1, epoch)
#         W.add_scalar('psnr', psnr, epoch)
#         W.add_scalar('ssim', ssim, epoch)
#         # torch.save(modelG, save_path + '/modelG1.pk')
#         # torch.save(modelP11, save_path + '/modelP11_c_G.pk')
#         # torch.save(modelR11, save_path + '/modelR11_c_G.pk')
#         torch.save(modelR_P, save_path + '/modelR_P_G1.pk')
#         torch.save(modelG_c, save_path + '/modelG_c_.pk')
#
# W.close()






# ########  training DU 3slice
# for epoch in range(num_epoches):
#     ####adjust learning_rate
#     if epoch % 300 == 0:
#         for param_group in optimizerDU.param_groups:
#             param_group['lr'] *= 0.1
#
#     for i,(T1_74, T1_75, T1_76, T2_74, T2_75, T2_76, UnderT2_74, UnderT2_75, UnderT2_76) in enumerate(data_loader):
#         T1_74 = T1_74.view(-1,1,240,240)
#         T1_74 = T1_74.cuda()
#         T1_75 = T1_75.view(-1, 1, 240, 240)
#         T1_75 = T1_75.cuda()
#         T1_76 = T1_76.view(-1, 1, 240, 240)
#         T1_76 = T1_76.cuda()
#         T2_74 = T2_74.view(-1,1,240,240)
#         T2_74 = T2_74.cuda()
#         T2_75 = T2_75.view(-1, 1, 240, 240)
#         T2_75 = T2_75.cuda()
#         T2_76 = T2_76.view(-1, 1, 240, 240)
#         T2_76 = T2_76.cuda()
#         UnderT2_74 = UnderT2_74.view(-1,1,240,240)
#         UnderT2_74 = UnderT2_74.cuda()
#         UnderT2_75 = UnderT2_75.view(-1, 1, 240, 240)
#         UnderT2_75 = UnderT2_75.cuda()
#         UnderT2_76 = UnderT2_76.view(-1, 1, 240, 240)
#         UnderT2_76 = UnderT2_76.cuda()
#
#         T1 = torch.cat((T1_74, T1_75, T1_76), 1)
#         T2 = torch.cat((T2_74, T2_75, T2_76), 1)
#         under_T2 = torch.cat((UnderT2_74, UnderT2_75, UnderT2_76), 1)
#
#         T2_Gen = modelDU(T1, under_T2)
#         optimizerDU.zero_grad()
#         l_G = loss(T2_Gen, T2)
#         l_G.backward()
#         optimizerDU.step()
#
#         if i % 10 == 0:
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))
#
#     if epoch % test_interval ==0 :
#         psnr = 0
#         ssim = 0
#         l_G_test = 0
#
#         with torch.no_grad():
#             for T1_74_test, T1_75_test, T1_76_test, T2_74_test, T2_75_test, T2_76_test, UnderT2_74_test, UnderT2_75_test, UnderT2_76_test in test_loader:
#                 T1_74_test = T1_74_test.view(-1, 1, 240, 240)
#                 T1_74_test = T1_74_test.cuda()
#                 T1_75_test = T1_75_test.view(-1, 1, 240, 240)
#                 T1_75_test = T1_75_test.cuda()
#                 T1_76_test = T1_76_test.view(-1, 1, 240, 240)
#                 T1_76_test = T1_76_test.cuda()
#                 T2_74_test = T2_74_test.view(-1, 1, 240, 240)
#                 T2_74_test = T2_74_test.cuda()
#                 T2_75_test = T2_75_test.view(-1, 1, 240, 240)
#                 T2_75_test = T2_75_test.cuda()
#                 T2_76_test = T2_76_test.view(-1, 1, 240, 240)
#                 T2_76_test = T2_76_test.cuda()
#                 UnderT2_74_test = UnderT2_74_test.view(-1, 1, 240, 240)
#                 UnderT2_74_test = UnderT2_74_test.cuda()
#                 UnderT2_75_test = UnderT2_75_test.view(-1, 1, 240, 240)
#                 UnderT2_75_test = UnderT2_75_test.cuda()
#                 UnderT2_76_test = UnderT2_76_test.view(-1, 1, 240, 240)
#                 UnderT2_76_test = UnderT2_76_test.cuda()
#
#                 T1_test = torch.cat((T1_74_test, T1_75_test, T1_76_test), 1)
#                 T2_test = torch.cat((T2_74_test, T2_75_test, T2_76_test), 1)
#                 under_T2_test = torch.cat((UnderT2_74_test, UnderT2_75_test, UnderT2_76_test), 1)
#
#                 T2_Gen_test = modelDU(T1_test, under_T2_test)
#
#                 l_G_test +=loss(T2_Gen_test, T2_test)
#                 T2_G_test_ = T2_Gen_test.cpu()
#                 T2_G_test_ = T2_G_test_.data.numpy().squeeze()
#                 T2_test_ = T2_test.cpu().numpy().squeeze()
#                 psnr += compare_psnr(T2_test_[0, :, :], T2_G_test_[0, :, :], data_range=T2_test_[0, :, :].max())
#                 psnr += compare_psnr(T2_test_[1, :, :], T2_G_test_[1, :, :], data_range=T2_test_[1, :, :].max())
#                 psnr += compare_psnr(T2_test_[2, :, :], T2_G_test_[2, :, :], data_range=T2_test_[2, :, :].max())
#                 ssim += compare_ssim(T2_test_[0, :, :], T2_G_test_[0, :, :], data_range=T2_test_[0, :, :].max())
#                 ssim += compare_ssim(T2_test_[1, :, :], T2_G_test_[1, :, :], data_range=T2_test_[1, :, :].max())
#                 ssim += compare_ssim(T2_test_[2, :, :], T2_G_test_[2, :, :], data_range=T2_test_[2, :, :].max())
#
#             psnr /= 3*len(test_loader)
#             ssim /= 3*len(test_loader)
#             l_G_test /= len(test_loader)
#
#         print('epoch:{:d},\tz_loss:{:.4f},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(epoch,100*l_G_test.item(),psnr.item(),ssim.item()))
#         torch.save(modelDU, save_path + '/modelDU.pk')



# ######  training G Res_net 迭代
# for epoch in range(num_epoches):
#     ####adjust learning_rate
#     # if epoch == 80:
#     #     for param_group in optimizerG_7.param_groups:
#     #         param_group['lr'] *= 0.1
#     if epoch % 300  == 0:
#         for param_group in optimizerG_16_6.param_groups:
#             param_group['lr'] *= 0.1
#         # for param_group in optimizerG_1.param_groups:
#         #     param_group['lr'] *= 0.1
#
#     # l_train = 0
#     for i,(T1,T2,under_T2,under_T2_K) in enumerate(data_loader):
#         T1 = T1.view(-1,1,240,240)
#         T1 = T1.cuda()
#         T2 = T2.view(-1,1,240,240)
#         T2 = T2.cuda()
#         under_T2 = under_T2.view(-1, 1, 240, 240)
#         under_T2 = under_T2.cuda()
#         under_T2_K = under_T2_K.view(-1, 2, 240, 240)
#
#         # with torch.no_grad():
#         #     T2_Gen = modelG_0(T1, under_T2)
#         #     T2_Gen = T2_Gen.detach()
#         #     z_1 = modelupdata_z(T2_Gen, under_T2_K)
#         #     T2_1 = modelG_1(T1, z_1)
#         #     z_2 = modelupdata_z(T2_1, under_T2_K)
#         #     T2_2 = modelG_2(T1, z_2)
#         #     z_3 = modelupdata_z(T2_2, under_T2_K)
#         #     T2_3 = modelG_3(T1, z_3)
#         #     z_4 = modelupdata_z(T2_3, under_T2_K)
#         #     T2_4 = modelG_4(T1, z_4)
#         #     z_5 = modelupdata_z(T2_4, under_T2_K)
#         #     T2_5 = modelG_5(T1, z_5)
#         #     z_6 = modelupdata_z(T2_5, under_T2_K)
#         #     T2_6 = modelG_6(T1, z_6)
#         #     z_7 = modelupdata_z(T2_6, under_T2_K)
#         #     T2_7 = modelG_7(T1, z_7)
#         # z_8 = modelupdata_z(T2_7, under_T2_K)
#         # T2_8 = modelG_8(T1, z_8)
#         with torch.no_grad():
#             T2_Gen = modelG_16_0(T1, under_T2)
#             T2_Gen = T2_Gen.detach()
#             z_1 = modelupdata_z(T2_Gen, under_T2_K)
#             T2_1 = modelG_16_1(T1, z_1)
#             z_2 = modelupdata_z(T2_1, under_T2_K)
#             T2_2 = modelG_16_2(T1, z_2)
#             z_3 = modelupdata_z(T2_2, under_T2_K)
#             T2_3 = modelG_16_3(T1, z_3)
#             z_4 = modelupdata_z(T2_3, under_T2_K)
#             T2_4 = modelG_16_4(T1, z_4)
#             z_5 = modelupdata_z(T2_4, under_T2_K)
#             T2_5 = modelG_16_5(T1, z_5)
#             z_6 = modelupdata_z(T2_5, under_T2_K)
#             T2_6 = modelG_16_6(T1, z_6)
#             z_7 = modelupdata_z(T2_6, under_T2_K)
#             T2_7 = modelG_16_7(T1, z_7)
#         z_8 = modelupdata_z(T2_7, under_T2_K)
#         T2_8 = modelG_16_8(T1, z_8)
#         optimizerG_16_6.zero_grad()
#         l_G = loss(T2_8,T2)
#         l_G.backward()
#         optimizerG_16_6.step()
#
#         if i % 10 == 0:
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))
#         # l_train += l_G
#
#     # Writer.add_scalar('train_loss', 100*l_train/len(data_loader), epoch)
#
#     if epoch % test_interval ==0 :
#         psnr = 0
#         ssim = 0
#         l_G_test = 0
#
#         with torch.no_grad():
#             for T1_test,T2_test,under_T2_test,under_T2_test_K in test_loader:
#                 T1_test = T1_test.view(-1,1,240,240)
#                 T1_test = T1_test.cuda()
#                 T2_test = T2_test.view(-1,1,240,240)
#                 T2_test = T2_test.cuda()
#                 under_T2_test = under_T2_test.view(-1, 1, 240, 240)
#                 under_T2_test = under_T2_test.cuda()
#                 under_T2_test_K = under_T2_test_K.view(-1, 2, 240, 240)
#
#                 # T2_G_test = modelG_0(T1_test, under_T2_test)
#                 # z_1_test = modelupdata_z(T2_G_test, under_T2_test_K)
#                 # T2_1_test = modelG_1(T1_test, z_1_test)
#                 # z_2_test = modelupdata_z(T2_1_test, under_T2_test_K)
#                 # T2_2_test = modelG_2(T1_test, z_2_test)
#                 # z_3_test = modelupdata_z(T2_2_test, under_T2_test_K)
#                 # T2_3_test = modelG_3(T1_test, z_3_test)
#                 # z_4_test = modelupdata_z(T2_3_test, under_T2_test_K)
#                 # T2_4_test = modelG_4(T1_test, z_4_test)
#                 # z_5_test = modelupdata_z(T2_4_test, under_T2_test_K)
#                 # T2_5_test = modelG_5(T1_test, z_5_test)
#                 # z_6_test = modelupdata_z(T2_5_test, under_T2_test_K)
#                 # T2_6_test = modelG_6(T1_test, z_6_test)
#                 # z_7_test = modelupdata_z(T2_6_test, under_T2_test_K)
#                 # T2_7_test = modelG_7(T1_test, z_7_test)
#                 # z_8_test = modelupdata_z(T2_7_test, under_T2_test_K)
#                 # T2_8_test = modelG_8(T1_test, z_8_test)
#                 T2_G_test = modelG_16_0(T1_test, under_T2_test)
#                 z_1_test = modelupdata_z(T2_G_test, under_T2_test_K)
#                 T2_1_test = modelG_16_1(T1_test, z_1_test)
#                 z_2_test = modelupdata_z(T2_1_test, under_T2_test_K)
#                 T2_2_test = modelG_16_2(T1_test, z_2_test)
#                 z_3_test = modelupdata_z(T2_2_test, under_T2_test_K)
#                 T2_3_test = modelG_16_3(T1_test, z_3_test)
#                 z_4_test = modelupdata_z(T2_3_test, under_T2_test_K)
#                 T2_4_test = modelG_16_4(T1_test, z_4_test)
#                 z_5_test = modelupdata_z(T2_4_test, under_T2_test_K)
#                 T2_5_test = modelG_16_5(T1_test, z_5_test)
#                 z_6_test = modelupdata_z(T2_5_test, under_T2_test_K)
#                 T2_6_test = modelG_16_6(T1_test, z_6_test)
#                 z_7_test = modelupdata_z(T2_6_test, under_T2_test_K)
#                 T2_7_test = modelG_16_7(T1_test, z_7_test)
#                 z_8_test = modelupdata_z(T2_7_test, under_T2_test_K)
#                 T2_8_test = modelG_16_8(T1_test, z_8_test)
#
#                 l_G_test +=loss(T2_8_test, T2_test)
#                 T2_1_test_ = T2_8_test.cpu()
#                 T2_1_test_ = T2_1_test_.data.numpy().squeeze()
#                 T2_test_ = T2_test.cpu().numpy().squeeze()
#                 psnr += compare_psnr(T2_test_, T2_1_test_, data_range=T2_test_.max())
#                 ssim += compare_ssim(T2_test_, T2_1_test_, data_range=T2_test_.max())
#
#             psnr /= len(test_loader)
#             ssim /= len(test_loader)
#             l_G_test /= len(test_loader)
#
#         print('epoch:{:d},\tz_loss:{:.4f},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(epoch,100*l_G_test.item(),psnr.item(),ssim.item()))
#         # writer.add_scalar('test_loss', 100*l_G_test, epoch)
#         # writer.add_scalar('psnr', psnr, epoch)
#         # writer.add_scalar('ssim', ssim, epoch)
#         # torch.save(modelG_0, save_path + '/modelG_10.pk')
#         torch.save(modelG_16_8, save_path + '/modelG_16_8_' + str(epoch) + '.pk')


# ########  training Dense_Unet 迭代
# for epoch in range(num_epoches):
#     ###adjust learning_rate
#     # if epoch == 100:
#     #     for param_group in optimizerDU_1.param_groups:
#     #         param_group['lr'] *= 0.1
#     if epoch % 150 == 0:
#         for param_group in optimizerDU_8.param_groups:
#             param_group['lr'] *= 0.1
#
#     l_train = 0
#     for i,(T1,T2,under_T2,under_T2_K) in enumerate(data_loader):
#         T1 = T1.view(-1,1,240,240)
#         T1 = T1.cuda()
#         T2 = T2.view(-1,1,240,240)
#         T2 = T2.cuda()
#         under_T2 = under_T2.view(-1, 1, 240, 240)
#         under_T2 = under_T2.cuda()
#         under_T2_K = under_T2_K.view(-1, 2, 240, 240)
#
#         with torch.no_grad():
#             T2_0 = modelDU_0(T1, under_T2)
#             z_1 = modelupdata_z(T2_0, under_T2_K)
#             T2_1 = modelDU_1(T1, z_1)
#             z_2 = modelupdata_z(T2_1, under_T2_K)
#             T2_2 = modelDU_2(T1, z_2)
#             z_3 = modelupdata_z(T2_2, under_T2_K)
#             T2_3 = modelDU_3(T1, z_3)
#             z_4 = modelupdata_z(T2_3, under_T2_K)
#             T2_4 = modelDU_4(T1, z_4)
#             z_5 = modelupdata_z(T2_4, under_T2_K)
#             T2_5 = modelDU_5(T1, z_5)
#             z_6 = modelupdata_z(T2_5, under_T2_K)
#             T2_6 = modelDU_6(T1, z_6)
#             z_7 = modelupdata_z(T2_6, under_T2_K)
#             T2_7 = modelDU_7(T1, z_7)
#         z_8 = modelupdata_z(T2_7, under_T2_K)
#         T2_8 = modelDU_8(T1, z_8)
#         optimizerDU_8.zero_grad()
#         l_G = loss(T2_8,T2)
#         l_G.backward()
#         optimizerDU_8.step()
#
#         if i % 10 == 0:
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))
#         l_train += l_G
#
#     W.add_scalar('train_loss', 100*l_train/len(data_loader), epoch)
#
#
#     if epoch % test_interval ==0 :
#         psnr_z = 0
#         ssim_z = 0
#         psnr = 0
#         ssim = 0
#         l_G_test = 0
#
#         with torch.no_grad():
#             for T1_test,T2_test,under_T2_test,under_T2_test_K in test_loader:
#                 T1_test = T1_test.view(-1,1,240,240)
#                 T1_test = T1_test.cuda()
#                 T2_test = T2_test.view(-1,1,240,240)
#                 T2_test = T2_test.cuda()
#                 under_T2_test = under_T2_test.view(-1, 1, 240, 240)
#                 under_T2_test = under_T2_test.cuda()
#                 under_T2_test_K = under_T2_test_K.view(-1, 2, 240, 240)
#
#                 T2_0_test = modelDU_0(T1_test, under_T2_test)
#                 z_1_test = modelupdata_z(T2_0_test, under_T2_test_K)
#                 T2_1_test = modelDU_1(T1_test, z_1_test)
#                 z_2_test = modelupdata_z(T2_1_test, under_T2_test_K)
#                 T2_2_test = modelDU_2(T1_test, z_2_test)
#                 z_3_test = modelupdata_z(T2_2_test, under_T2_test_K)
#                 T2_3_test = modelDU_3(T1_test, z_3_test)
#                 z_4_test = modelupdata_z(T2_3_test, under_T2_test_K)
#                 T2_4_test = modelDU_4(T1_test, z_4_test)
#                 z_5_test = modelupdata_z(T2_4_test, under_T2_test_K)
#                 T2_5_test = modelDU_5(T1_test, z_5_test)
#                 z_6_test = modelupdata_z(T2_5_test, under_T2_test_K)
#                 T2_6_test = modelDU_6(T1_test, z_6_test)
#                 z_7_test = modelupdata_z(T2_6_test, under_T2_test_K)
#                 T2_7_test = modelDU_7(T1_test, z_7_test)
#                 z_8_test = modelupdata_z(T2_7_test, under_T2_test_K)
#                 T2_8_test = modelDU_8(T1_test, z_8_test)
#                 l_G_test +=loss(T2_8_test, T2_test)
#                 z_8_test_ = z_8_test.cpu()
#                 z_8_test_ = z_8_test_.data.numpy().squeeze()
#                 T2_4_test_ = T2_8_test.cpu()
#                 T2_4_test_ = T2_4_test_.data.numpy().squeeze()
#                 T2_test_ = T2_test.cpu().numpy().squeeze()
#                 psnr_z += compare_psnr(T2_test_, z_8_test_, data_range=T2_test_.max())
#                 ssim_z += compare_ssim(T2_test_, z_8_test_, data_range=T2_test_.max())
#                 psnr += compare_psnr(T2_test_, T2_4_test_, data_range=T2_test_.max())
#                 ssim += compare_ssim(T2_test_, T2_4_test_, data_range=T2_test_.max())
#
#             psnr_z /= len(test_loader)
#             ssim_z /= len(test_loader)
#             psnr /= len(test_loader)
#             ssim /= len(test_loader)
#             l_G_test /= len(test_loader)
#
#         print(
#             'epoch:{:d},\tz_loss:{:.4f},\ttest_psnr_z:{:.4f},\ttest_ssim_z:{:.4f},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(
#                 epoch, 100 * l_G_test.item(), psnr_z.item(), ssim_z.item(), psnr.item(), ssim.item()))
#         W.add_scalar('test_loss', 100*l_G_test, epoch)
#         W.add_scalar('psnr', psnr, epoch)
#         W.add_scalar('ssim', ssim, epoch)
#         torch.save(modelDU_8, save_path + '/modelDU_8_' + str(epoch) + '.pk')
# W.close()



# ########  training Res_attention_net 迭代
# for epoch in range(num_epoches):
#     ###adjust learning_rate
#     if epoch == 80:
#         for param_group in optimizerGA2.param_groups:
#             param_group['lr'] *= 0.1
#     if epoch % 50 == 0:
#         for param_group in optimizerGA2.param_groups:
#             param_group['lr'] *= 0.1
#
#     for i,(T1,T2,under_T2) in enumerate(train_loader):
#         T1 = T1.view(-1,1,240,240)
#         T1 = T1.cuda()
#         T2 = T2.view(-1,1,240,240)
#         T2 = T2.cuda()
#         under_T2 = under_T2.view(-1, 1, 240, 240)
#         under_T2 = under_T2.cuda()
#         # under_T2_K = under_T2_K.view(-1, 2, 240, 240)
#
#         # with torch.no_grad():
#         T2_0 = modelGA2_0(T1, under_T2)
#         #     z_1 = modelupdata_z(T2_0, under_T2_K)
#         # T2_1 = modelFFA_res_1(T1, z_1)
#         optimizerGA2.zero_grad()
#         l_G = loss(T2_0,T2)
#         l_G.backward()
#         optimizerGA2.step()
#
#         if i % 10 == 0:
#             print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))
#
#     if epoch > 35:
#         if epoch % test_interval == 0:
#             # psnr_z = 0
#             # ssim_z = 0
#             psnr = 0
#             ssim = 0
#             # l_G_test = 0
#
#             with torch.no_grad():
#                 for j, (T1_test, T2_test, under_T2_test) in enumerate(test_loader):
#                     T1_test = T1_test.view(-1, 1, 240, 240, 115)
#                     T1_test = T1_test.cuda()
#                     T2_test = T2_test.view(-1, 1, 240, 240, 115)
#                     T2_test = T2_test.cuda()
#                     under_T2_test = under_T2_test.view(-1, 1, 240, 240, 115)
#                     under_T2_test = under_T2_test.cuda()
#                     # under_T2_test_K = under_T2_test_K.view(-1, 2, 240, 240)
#                     T2_0_test = torch.zeros(1, 1, 240, 240, 115)
#                     for k in range(115):
#                         T2_0_test[:, :, :, :, k] = modelGA2_0(T1_test[:, :, :, :, k], under_T2_test[:, :, :, :, k])
#
#                     # z_1_test = modelupdata_z(T2_0_test, under_T2_test_K)
#                     # T2_1_test = modelFFA_res_1(T1_test, z_1_test)
#                     # l_G_test +=loss(T2_0_test, T2_test)
#                     # z_0_test_ = z_0_test.cpu()
#                     # z_8_test_ = z_8_test_.data.numpy().squeeze()
#                     T2_0_test_ = T2_0_test.cpu()
#                     T2_0_test_ = T2_0_test_.data.numpy().squeeze()
#                     T2_test_ = T2_test.cpu().numpy().squeeze()
#                     # psnr_z += compare_psnr(T2_test_, z_8_test_, data_range=T2_test_.max())
#                     # ssim_z += compare_ssim(T2_test_, z_8_test_, data_range=T2_test_.max())
#                     psnr_single = compare_psnr(T2_test_, T2_0_test_, data_range=T2_test_.max())
#                     ssim_single = compare_ssim(T2_test_, T2_0_test_, data_range=T2_test_.max())
#                     print('epoch:{:d},\tstep:{:d},\ttest_psnr_single:{:.4f},\ttest_ssim_single:{:.4f}'.format(epoch, j,
#                                                                                                               psnr_single.item(),
#                                                                                                               ssim_single.item()))
#                     psnr += psnr_single
#                     ssim += ssim_single
#
#                 # psnr_z /= len(test_loader)
#                 # ssim_z /= len(test_loader)
#                 psnr /= len(test_loader)
#                 ssim /= len(test_loader)
#                 # l_G_test /= len(test_loader)
#             print('epoch:{:d},\ttest_psnr:{:.4f},\ttest_ssim:{:.4f}'.format(epoch, psnr.item(), ssim.item()))
#
#     torch.save(modelGA2_0, save_path + '/modelGA2_0_' + str(epoch) + '.pk')


########  training FFA_res_single_net 迭代
for epoch in range(num_epoches):
    ####adjust learning_rate
    if epoch == 60:
        for param_group in optimizerFFA_res_0.param_groups:
            param_group['lr'] *= 0.1
    if epoch == 52:
        for param_group in optimizerFFA_res_0.param_groups:
            param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_1.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_2.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_3.param_groups:
        #     param_group['lr'] *= 0.1
    if epoch % 36 == 0:
        for param_group in optimizerFFA_res_0.param_groups:
            param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_1.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_2.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_3.param_groups:
        #     param_group['lr'] *= 0.1

    for i, (T1, T2, under_T2, under_T2_K) in enumerate(train_loader):
        T1 = T1.view(-1, 1, 240, 240)
        T1 = T1.cuda()
        T2 = T2.view(-1, 1, 240, 240)
        T2 = T2.cuda()
        under_T2 = under_T2.view(-1, 1, 240, 240)
        under_T2 = under_T2.cuda()
        # under_T2_K = under_T2_K.view(-1, 1, 240, 240, 2)
        # under_T2_K = under_T2_K.cuda()

        # with torch.no_grad():
        #     T2_0 = modelFFA_res_0(T1, under_T2)
        #     z_1 = modelupdata_x2(T2_0, under_T2_K)
        T2_0 = modelFFA_res_0(T1, under_T2)
        # z_1 = modelupdata_x2(T2_0, under_T2_K)
        # T2_1 = modelFFA_res_1(T1, z_1)
        # z_2 = modelupdata_x2(T2_1, under_T2_K)
        # T2_2 = modelFFA_res_2(T1, z_2)
        # z_3 = modelupdata_x2(T2_2, under_T2_K)
        # T2_3 = modelFFA_res_3(T1, z_3)
        # optimizerFFA_res_3.zero_grad()
        # optimizerFFA_res_2.zero_grad()
        # optimizerFFA_res_1.zero_grad()
        optimizerFFA_res_0.zero_grad()
        l_G = loss(T2_0, T2)
        l_G.backward()
        # optimizerFFA_res_3.step()
        # optimizerFFA_res_2.step()
        # optimizerFFA_res_1.step()
        optimizerFFA_res_0.step()

        if i % 1000 == 0:
            print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))

    if epoch > 49:
        if epoch % test_interval == 0:
            # psnr_z = 0
            # nrmse_z = 0
            psnr = 0
            nrmse = 0
            with torch.no_grad():
                for j, (T1_test, T2_test, under_T2_test, under_T2_test_K) in enumerate(test_loader):
                    T1_test = T1_test.view(-1, 1, 240, 240, 115)
                    T1_test = T1_test.cuda()
                    T2_test = T2_test.view(-1, 1, 240, 240, 115)
                    T2_test = T2_test.cuda()
                    under_T2_test = under_T2_test.view(-1, 1, 240, 240, 115)
                    under_T2_test = under_T2_test.cuda()
                    # under_T2_test_K = under_T2_test_K.view(-1, 1, 240, 240, 115, 2)
                    # under_T2_test_K = under_T2_test_K.cuda()
                    T2_0_test = torch.zeros(1, 1, 240, 240, 115)
                    T2_0_test = T2_0_test.cuda()
                    # z_1_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_1_test = z_1_test.cuda()
                    # T2_1_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_1_test = T2_1_test.cuda()
                    # z_2_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_2_test = z_2_test.cuda()
                    # T2_2_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_2_test = T2_2_test.cuda()
                    # z_3_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_3_test = z_3_test.cuda()
                    # T2_3_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_3_test = T2_3_test.cuda()
                    # z_4_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_4_test = z_4_test.cuda()
                    # T2_4_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_4_test = T2_4_test.cuda()
                    # z_5_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_5_test = z_5_test.cuda()
                    # T2_5_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_5_test = T2_5_test.cuda()
                    # z_6_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_6_test = z_6_test.cuda()
                    # T2_6_test = torch.zeros(1, 1, 240, 240, 115)
                    # T2_6_test = T2_6_test.cuda()
                    # z_7_test = torch.zeros(1, 1, 240, 240, 115)
                    # z_7_test = z_7_test.cuda()

                    for k in range(115):
                        T2_0_test[:, :, :, :, k] = modelFFA_res_0(T1_test[:, :, :, :, k], under_T2_test[:, :, :, :, k])
                        # z_1_test[:, :, :, :, k] = modelupdata_x2(T2_0_test[:, :, :, :, k],
                        #                                          under_T2_test_K[:, :, :, :, k, :])
                        # T2_1_test[:, :, :, :, k] = modelFFA_res_1(T1_test[:, :, :, :, k], z_1_test[:, :, :, :, k])
                        # z_2_test[:, :, :, :, k] = modelupdata_x2(T2_1_test[:, :, :, :, k],
                        #                                          under_T2_test_K[:, :, :, :, k, :])
                        # T2_2_test[:, :, :, :, k] = modelFFA_res_2(T1_test[:, :, :, :, k], z_2_test[:, :, :, :, k])
                        # z_3_test[:, :, :, :, k] = modelupdata_x2(T2_2_test[:, :, :, :, k],
                        #                                          under_T2_test_K[:, :, :, :, k, :])
                        # T2_3_test[:, :, :, :, k] = modelFFA_res_3(T1_test[:, :, :, :, k], z_3_test[:, :, :, :, k])
                        # z_4_test[:, :, :, :, k] = modelupdata_x2(T2_3_test[:, :, :, :, k],
                        #                                          under_T2_test_K[:, :, :, :, k, :])

                    # MAE_test = MAE(T2_0_test, T2_test)
                    # mae += MAE_test
                    T2_0_test_ = T2_0_test.cpu()
                    T2_0_test_ = T2_0_test_.data.numpy().squeeze()
                    # z_2_test_ = z_4_test.cpu()
                    # z_2_test_ = z_2_test_.data.numpy().squeeze()
                    T2_test_ = T2_test.cpu().numpy().squeeze()
                    # psnr_z += compare_psnr(T2_test_, z_2_test_, data_range=T2_test_.max())
                    # nrmse_z += compare_nrmse(T2_test_, z_2_test_)
                    psnr_singal = compare_psnr(T2_test_, T2_0_test_, data_range=T2_test_.max())
                    nrmse_singal = compare_nrmse(T2_test_, T2_0_test_)
                    print(
                        'epoch:{:d},\tstep:{:d},\ttest_psnr_single:{:.4f},\ttest_nrmse_single:{:.4f}'.format(
                            epoch, j,
                            psnr_singal.item(),
                            nrmse_singal.item()))
                    psnr += psnr_singal
                    nrmse += nrmse_singal

                # psnr_z /= len(test_loader)
                # nrmse_z /= len(test_loader)
                psnr /= len(test_loader)
                nrmse /= len(test_loader)
                print('epoch:{:d},\ttest_psnr:{:.4f},\ttest_nrmse:{:.4f}'.format(epoch, psnr.item(), nrmse.item()))
                # print('epoch:{:d},\ttest_psnrz:{:.4f},\ttest_nrmse:{:.4f}'.format(epoch, psnr_z.item(), nrmse_z.item()))

    torch.save(modelFFA_res_0, save_path + '/modelFFA_res16_01_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_1, save_path + '/modelFFA_res_NoA_31_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_2, save_path + '/modelFFA_res_NoA_32_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_3, save_path + '/modelFFA_res_NoA_33_' + str(epoch) + '.pk')

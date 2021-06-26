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
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

save_path = 'snapshot'
if not os.path.exists(save_path):
    os.mkdir(save_path)

######### hyper-parameters
batch_size = 2
learning_rate = 1e-4
weight_decay = 1e-5
num_epoches = 90
test_interval = 10

num_chanel_Res = 16
num_blocks_Res = 9

num_chanel_Dense = 16
depth_Dense = 4

gps=3
blocks=19

Mask = sio.loadmat('Mask_8.mat')['Mask']
Mask = torch.Tensor(Mask)

train_path = "sliceT1.txt"
test_path = "T1_mat/Test.txt"

######### data_loader and network
train_loader = torch.utils.data.DataLoader(load_data.Train_List(train_path),batch_size=4,shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(load_data.Test_List(test_path),batch_size=1,shuffle=True, num_workers=1,drop_last=True)

modelFFA_res_0 = network.FFA_res().cuda()

# modelFFA_res_0 = torch.load(save_path+'/modelFFA_res32_30_5.pk', map_location='cuda:0')
# modelFFA_res_1 = torch.load(save_path+'/modelFFA_res32_31_5.pk', map_location='cuda:0')
# modelFFA_res_2 = torch.load(save_path+'/modelFFA_res32_32_5.pk', map_location='cuda:0')
# modelFFA_res_3 = torch.load(save_path+'/modelFFA_res32_33_5.pk', map_location='cuda:0')

modelupdata_x2 = network.updata_x2(Mask).cuda()

######optimizer and loss
optimizerFFA_res_0 = torch.optim.Adam(modelFFA_res_0.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_1 = torch.optim.Adam(modelFFA_res_1.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_2 = torch.optim.Adam(modelFFA_res_2.parameters(),lr=learning_rate,weight_decay=weight_decay)
# optimizerFFA_res_3 = torch.optim.Adam(modelFFA_res_3.parameters(),lr=learning_rate,weight_decay=weight_decay)

loss = nn.MSELoss()


#######  training FFA_res_net
for epoch in range(num_epoches):
    ###adjust learning_rate
    if epoch % 70 == 0 or epoch % 80 == 0:
        # for param_group in optimizerFFA_res_3.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_2.param_groups:
        #     param_group['lr'] *= 0.1
        # for param_group in optimizerFFA_res_1.param_groups:
        #     param_group['lr'] *= 0.1
        for param_group in optimizerFFA_res_0.param_groups:
            param_group['lr'] *= 0.1

    for i, (T1, T2, under_T2, under_T2_K) in enumerate(train_loader):
        T1 = T1.view(-1, 1, 240, 240)
        T1 = T1.cuda()
        T2 = T2.view(-1, 1, 240, 240)
        T2 = T2.cuda()
        under_T2 = under_T2.view(-1, 1, 240, 240)
        under_T2 = under_T2.cuda()
        under_T2_K = under_T2_K.view(-1, 1, 240, 240, 2)
        under_T2_K = under_T2_K.cuda()

        T2_0 = modelFFA_res_0(T1, under_T2)
        # z_1 = modelupdata_x2(T2_0, under_T2_K)
        # T2_1 = modelFFA_res_1(T1, z_1)
        # z_2 = modelupdata_x2(T2_1, under_T2_K)
        # T2_2 = modelFFA_res_2(T1, z_2)
        # z_3 = modelupdata_x2(T2_2, under_T2_K)
        # T2_3 = modelFFA_res_2(T1, z_3)

        # optimizerFFA_res_3.zero_grad()
        # optimizerFFA_res_2.zero_grad()
        # optimizerFFA_res_1.zero_grad()
        optimizerFFA_res_0.zero_grad()
        l_G = loss(T2_0, T2)   # or T2_1 or T2_2 or T2_3
        l_G.backward()
        # optimizerFFA_res_3.step()
        # optimizerFFA_res_2.step()
        # optimizerFFA_res_1.step()
        optimizerFFA_res_0.step()

        if i % 10 == 0:
            print('epoch:{:d},\tstep:{:d},\tloss:{:.4f}'.format(epoch,i,100*l_G.item()))


    if epoch % test_interval == 0:
        psnr_z = 0
        nrmse_z = 0
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
                under_T2_test_K = under_T2_test_K.view(-1, 1, 240, 240, 115, 2)
                under_T2_test_K = under_T2_test_K.cuda()
                T2_0_test = torch.zeros(1, 1, 240, 240, 115)
                T2_0_test = T2_0_test.cuda()
                z_1_test = torch.zeros(1, 1, 240, 240, 115)
                z_1_test = z_1_test.cuda()
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
                for k in range(115):
                    T2_0_test[:, :, :, :, k] = modelFFA_res_0(T1_test[:, :, :, :, k], under_T2_test[:, :, :, :, k])
                    z_1_test[:, :, :, :, k] = modelupdata_x2(T2_0_test[:, :, :, :, k],
                                                             under_T2_test_K[:, :, :, :, k, :])
                    # T2_1_test[:, :, :, :, k] = modelFFA_res_1(T1_test[:, :, :, :, k], z_1_test[:, :, :, :, k])
                    # z_2_test[:, :, :, :, k] = modelupdata_x2(T2_1_test[:, :, :, :, k],
                    #                                          under_T2_test_K[:, :, :, :, k, :])
                    # T2_2_test[:, :, :, :, k] = modelFFA_res_2(T1_test[:, :, :, :, k], z_2_test[:, :, :, :, k])
                    # z_3_test[:, :, :, :, k] = modelupdata_x2(T2_2_test[:, :, :, :, k],
                    #                                          under_T2_test_K[:, :, :, :, k, :])
                    # T2_3_test[:, :, :, :, k] = modelFFA_res_3(T1_test[:, :, :, :, k], z_3_test[:, :, :, :, k])
                    # z_4_test[:, :, :, :, k] = modelupdata_x2(T2_3_test[:, :, :, :, k],
                    #                                          under_T2_test_K[:, :, :, :, k, :])

                # l_G_test +=loss(T2_0_test, T2_test)
                z_1_test_ = z_1_test.cpu()
                z_1_test_ = z_1_test_.data.numpy().squeeze()
                T2_0_test_ = T2_0_test.cpu()
                T2_0_test_ = T2_0_test_.data.numpy().squeeze()
                T2_test_ = T2_test.cpu().numpy().squeeze()
                psnr_z += compare_psnr(T2_test_, z_1_test_, data_range=T2_test_.max())
                nrmse_z += compare_nrmse(T2_test_, z_1_test_)
                psnr_single = compare_psnr(T2_test_, T2_0_test_, data_range=T2_test_.max())
                nrmse_single = compare_nrmse(T2_test_, T2_0_test_)
                print('epoch:{:d},\tstep:{:d},\ttest_psnr_single:{:.4f},\ttest_nrmse_single:{:.4f}'.format(epoch, j,
                                                                                                          psnr_single.item(),
                                                                                                          nrmse_single.item()))
                psnr += psnr_single
                nrmse += nrmse_single

            psnr_z /= len(test_loader)
            nrmse_z /= len(test_loader)
            psnr /= len(test_loader)
            nrmse /= len(test_loader)
            # l_G_test /= len(test_loader)
            print('epoch:{:d},\ttest_psnr:{:.4f},\ttest_nrmse:{:.4f}'.format(epoch, psnr.item(), nrmse.item()))
            print('epoch:{:d},\ttest_psnrz:{:.4f},\ttest_nrmsez:{:.4f}'.format(epoch, psnr_z.item(), nrmse_z.item()))

    torch.save(modelFFA_res_0, save_path + '/modelFFA_res_0_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_1, save_path + '//modelFFA_res_1_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_2, save_path + '//modelFFA_res_2_' + str(epoch) + '.pk')
    # torch.save(modelFFA_res_3, save_path + '//modelFFA_res_3_' + str(epoch) + '.pk')

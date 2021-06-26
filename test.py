import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio
import load_data
import network
import os
from skimage.measure import compare_psnr, compare_ssim

######### hyper-parameters
batch_size = 1
num_chanel_Res = 16
num_blocks_Res = 9

Mask = sio.loadmat('Mask_8.mat')['Mask']

save_path = 'snapshot'
if not os.path.exists(save_path):
    os.mkdir(save_path)

train_path = "sliceT1.txt"
test_path = "T1_mat/Test.txt"

######### data_loader and network
test_loader = torch.utils.data.DataLoader(load_data.Test_List(test_path),batch_size=1,shuffle=False, num_workers=1,drop_last=True)
# modelG_0 = network.Generator_cat(num_blocks_Res,num_chanel_Res).cuda()
modelG_0 = torch.load(save_path+'/modelFFA_res_0.pk')
modelG_1 = torch.load(save_path+'/modelFFA_res_1.pk')
modelG_2 = torch.load(save_path+'/modelFFA_res_2.pk')
modelG_3 = torch.load(save_path+'/modelFFA_res_3.pk')

modelupdata_z = network.updata_z(Mask).cuda()


with torch.no_grad():
    psnr_under = 0
    ssim_under = 0
    psnr_x2_0 = 0
    ssim_x2_0 = 0
    psnr_z_0 = 0
    ssim_z_0 = 0
    psnr_x2_1 = 0
    ssim_x2_1 = 0
    psnr_z_1 = 0
    ssim_z_1 = 0
    psnr_x2_2 = 0
    ssim_x2_2 = 0
    psnr_z_2 = 0
    ssim_z_2 = 0
    psnr_x2_3 = 0
    ssim_x2_3 = 0
    psnr_z_3 = 0
    ssim_z_3 = 0

    for T1_test, T2_test, under_T2_test, under_T2_test_K in test_loader:
        T1_test = T1_test.view(-1, 1, 240, 240)
        T1_test = T1_test.cuda()
        T2_test = T2_test.view(-1, 1, 240, 240)
        T2_test = T2_test.cuda()
        T2_test_ = T2_test.cpu().numpy().squeeze()
        under_T2_test = under_T2_test.view(-1, 1, 240, 240)
        under_T2_test = under_T2_test.cuda()
        under_T2_test_K = under_T2_test_K.view(-1, 2, 240, 240)

        under_T2_test_ = under_T2_test.cpu().numpy().squeeze()
        psnr_under += compare_psnr(T2_test_, under_T2_test_, data_range=T2_test_.max())
        ssim_under += compare_ssim(T2_test_, under_T2_test_, data_range=T2_test_.max())

        x2_0 = modelG_0(T1_test, under_T2_test)
        x2_0_ = x2_0.cpu()
        x2_0_ = x2_0_.data.numpy().squeeze()
        psnr_x2_0 += compare_psnr(T2_test_, x2_0_, data_range=T2_test_.max())
        ssim_x2_0 += compare_ssim(T2_test_, x2_0_, data_range=T2_test_.max())

        z_0 = modelupdata_z(x2_0, under_T2_test_K)
        z_0_ = z_0.cpu()
        z_0_ = z_0_.data.numpy().squeeze()
        psnr_z_0 += compare_psnr(T2_test_, z_0_, data_range=T2_test_.max())
        ssim_z_0 += compare_ssim(T2_test_, z_0_, data_range=T2_test_.max())

        x2_1 = modelG_1(T1_test, z_0)
        x2_1_ = x2_1.cpu()
        x2_1_ = x2_1_.data.numpy().squeeze()
        psnr_x2_1 += compare_psnr(T2_test_, x2_1_, data_range=T2_test_.max())
        ssim_x2_1 += compare_ssim(T2_test_, x2_1_, data_range=T2_test_.max())

        z_1 = modelupdata_z(x2_1, under_T2_test_K)
        z_1_ = z_1.cpu()
        z_1_ = z_1_.data.numpy().squeeze()
        psnr_z_1 += compare_psnr(T2_test_, z_1_, data_range=T2_test_.max())
        ssim_z_1 += compare_ssim(T2_test_, z_1_, data_range=T2_test_.max())

        x2_2 = modelG_2(T1_test, z_1)
        x2_2_ = x2_2.cpu()
        x2_2_ = x2_2_.data.numpy().squeeze()
        psnr_x2_2 += compare_psnr(T2_test_, x2_2_, data_range=T2_test_.max())
        ssim_x2_2 += compare_ssim(T2_test_, x2_2_, data_range=T2_test_.max())

        z_2 = modelupdata_z(x2_2, under_T2_test_K)
        z_2_ = z_2.cpu()
        z_2_ = z_2_.data.numpy().squeeze()
        psnr_z_2 += compare_psnr(T2_test_, z_2_, data_range=T2_test_.max())
        ssim_z_2 += compare_ssim(T2_test_, z_2_, data_range=T2_test_.max())

        x2_3 = modelG_3(T1_test, z_2)
        x2_3_ = x2_3.cpu()
        x2_3_ = x2_3_.data.numpy().squeeze()
        psnr_x2_3 += compare_psnr(T2_test_, x2_3_, data_range=T2_test_.max())
        ssim_x2_3 += compare_ssim(T2_test_, x2_3_, data_range=T2_test_.max())

        z_3 = modelupdata_z(x2_3, under_T2_test_K)
        z_3_ = z_3.cpu()
        z_3_ = z_3_.data.numpy().squeeze()
        psnr_z_3 += compare_psnr(T2_test_, z_3_, data_range=T2_test_.max())
        ssim_z_3 += compare_ssim(T2_test_, z_3_, data_range=T2_test_.max())


    psnr_under /= len(test_loader)
    ssim_under /= len(test_loader)

    psnr_x2_0 /= len(test_loader)
    ssim_x2_0 /= len(test_loader)
    psnr_z_0 /= len(test_loader)
    ssim_z_0 /= len(test_loader)

    psnr_x2_1 /= len(test_loader)
    ssim_x2_1 /= len(test_loader)
    psnr_z_1 /= len(test_loader)
    ssim_z_1 /= len(test_loader)
    psnr_x2_2 /= len(test_loader)
    ssim_x2_2 /= len(test_loader)
    psnr_z_2 /= len(test_loader)
    ssim_z_2 /= len(test_loader)
    psnr_x2_3 /= len(test_loader)
    ssim_x2_3 /= len(test_loader)
    psnr_z_3 /= len(test_loader)
    ssim_z_3 /= len(test_loader)

print('psnr_under:{:.4f},\tssim_under:{:.4f}'.format(psnr_under.item(), ssim_under.item()))
print('psnr_x2_0:{:.4f},\tssim_x2_0:{:.4f},psnr_z_0:{:.4f},\tssim_z_0:{:.4f}'.format(psnr_x2_0.item(),ssim_x2_0.item(),psnr_z_0.item(),ssim_z_0.item()))
print('psnr_x2_1:{:.4f},\tssim_x2_1:{:.4f},psnr_z_1:{:.4f},\tssim_z_1:{:.4f}'.format(psnr_x2_1.item(),ssim_x2_1.item(),psnr_z_1.item(),ssim_z_1.item()))
print('psnr_x2_2:{:.4f},\tssim_x2_2:{:.4f},psnr_z_2:{:.4f},\tssim_z_2:{:.4f}'.format(psnr_x2_2.item(),ssim_x2_2.item(),psnr_z_2.item(),ssim_z_2.item()))
print('psnr_x2_3:{:.4f},\tssim_x2_3:{:.4f},psnr_z_3:{:.4f},\tssim_z_3:{:.4f}'.format(psnr_x2_3.item(),ssim_x2_3.item(),psnr_z_3.item(),ssim_z_3.item()))


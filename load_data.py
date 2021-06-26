import os
import scipy.io as sio
import random
import numpy as np

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class Train_List(Dataset):
    def __init__(self,path_T1,transform=None,mode='L'):
        self.path_T1 = path_T1
        self.transform = transform
        self.T1_list = open(self.path_T1).readlines()
        self.mode = mode

    def __getitem__(self, i):
        mat_T1 = self.T1_list[i].split()[0]
        file_path_list = mat_T1.split('/')

        file_path_list[-2] = 'sliceT2'
        mat_T2 = '/'.join(file_path_list)

        file_path_list[-2] = 'slice_under32'
        mat_UnderT2 = '/'.join(file_path_list)

        file_path_list[-2] = 'slice_under32_c'
        mat_Under_T2_K = '/'.join(file_path_list)

        T1 = sio.loadmat(mat_T1)["cmsli"]
        T1 = np.squeeze(T1)
        if self.transform is not None:
            T1 = Image.fromarray(T1).convert(mode=self.mode)
            T1 = self.transform(T1)
        else:
            T1 = T1 / 6000.
            T1 = torch.Tensor(T1)

        T2 = sio.loadmat(mat_T2)["cmsli"]
        T2 = np.squeeze(T2)
        if self.transform is not None:
            T2 = Image.fromarray(T2).convert(mode=self.mode)
            T2 = self.transform(T2)
        else:
            T2 = T2 / 6000.
            T2 = torch.Tensor(T2)

        UnderT2 = sio.loadmat(mat_UnderT2)["cmsli"]
        UnderT2 = np.squeeze(UnderT2)
        if self.transform is not None:
            UnderT2 = Image.fromarray(UnderT2).convert(mode=self.mode)
            UnderT2 = self.transform(UnderT2)
        else:
            UnderT2 = torch.Tensor(UnderT2)

        img_Under_T2_K = sio.loadmat(mat_Under_T2_K)["cmsli"]
        img_Under_T2_K_real = img_Under_T2_K.real
        img_Under_T2_K_real = torch.Tensor(img_Under_T2_K_real)
        img_Under_T2_K_real = img_Under_T2_K_real.view(1, 240, 240)
        img_Under_T2_K_imag = img_Under_T2_K.imag
        img_Under_T2_K_imag = torch.Tensor(img_Under_T2_K_imag)
        img_Under_T2_K_imag = img_Under_T2_K_imag.view(1, 240, 240)
        img_Under_T2_K = torch.stack((img_Under_T2_K_real, img_Under_T2_K_imag), 3)

        return T1, T2, UnderT2, img_Under_T2_K

    def __len__(self):
        return len(self.T1_list)



class Test_List(Dataset):
    def __init__(self,path_T1,transform=None,mode='L'):
        self.path_T1 = path_T1
        self.transform = transform
        self.T1_list = open(self.path_T1).readlines()
        self.mode = mode

    def __getitem__(self, i):
        mat_T1 = self.T1_list[i].split()[0]
        file_path_list = mat_T1.split('/')
        file_path_list[-3] = 'T2_mat'
        mat_T2 = '/'.join(file_path_list)

        file_path_list[-3] = 'UnderT2_32'
        mat_UnderT2 = '/'.join(file_path_list)

        file_path_list[-3] = 'UnderT2_32_c'
        mat_Under_T2_K = '/'.join(file_path_list)

        T1 = sio.loadmat(mat_T1)["cmsli"]
        T1 = np.squeeze(T1)
        if self.transform is not None:
            T1 = Image.fromarray(T1).convert(mode=self.mode)
            T1 = self.transform(T1)
        else:
            T1 = T1 / 6000.
            T1 = T1[:,:,20:135]
            T1 = torch.Tensor(T1)



        T2 = sio.loadmat(mat_T2)["cmsli"]
        T2 = np.squeeze(T2)
        if self.transform is not None:
            T2 = Image.fromarray(T2).convert(mode=self.mode)
            T2 = self.transform(T2)
        else:
            T2 = T2 / 6000.
            T2 = T2[:, :, 20:135]
            T2 = torch.Tensor(T2)


        UnderT2 = sio.loadmat(mat_UnderT2)["cmsli"]
        UnderT2 = np.squeeze(UnderT2)
        if self.transform is not None:
            UnderT2 = Image.fromarray(UnderT2).convert(mode=self.mode)
            UnderT2 = self.transform(UnderT2)
        else:
            UnderT2 = UnderT2[:, :, 20:135]
            UnderT2 = torch.Tensor(UnderT2)

        img_Under_T2_K = sio.loadmat(mat_Under_T2_K)["cmsli"]
        img_Under_T2_K = np.squeeze(img_Under_T2_K)
        img_Under_T2_K = img_Under_T2_K[:, :, 20:135]
        img_Under_T2_K_real = img_Under_T2_K.real
        img_Under_T2_K_real = torch.Tensor(img_Under_T2_K_real)
        img_Under_T2_K_real = img_Under_T2_K_real.view(1, 240, 240, 115)
        img_Under_T2_K_imag = img_Under_T2_K.imag
        img_Under_T2_K_imag = torch.Tensor(img_Under_T2_K_imag)
        img_Under_T2_K_imag = img_Under_T2_K_imag.view(1, 240, 240, 115)
        img_Under_T2_K = torch.stack((img_Under_T2_K_real, img_Under_T2_K_imag), 4)

        return T1, T2, UnderT2, img_Under_T2_K

    def __len__(self):
        return len(self.T1_list)

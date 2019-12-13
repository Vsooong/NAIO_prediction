# coding=utf-8
from __future__ import print_function, division
import torch
from numpy import *
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from PIL import Image
from Configs import cfg
import numpy as np


class naio_data(Dataset):
    def __init__(self, filepath):
        self.input = torch.as_tensor(np.load(os.path.join(filepath, 'train_input_5-1.npy')))
        self.input = self.input.permute(0, 1, 4, 2, 3).contiguous().float()
        self.target = torch.as_tensor(np.load(os.path.join(filepath, 'train_output_5-1.npy')))
        self.target = self.target.permute(0, 1, 4, 2, 3).contiguous().float()

    def __len__(self):
        return self.target.size(0)

    def __getitem__(self, idx):
        x, y = self.input[idx], self.target[idx]
        return x, y


if __name__ == '__main__':
    filepath = 'D:/Python_Project/NAIO_prediction/data_lj_nao/'
    data = naio_data(filepath)
    print(data.input.shape, data.target.shape)
    dataloader = DataLoader(data, batch_size=16, shuffle=False)

    # for minibatch in dataloader:
    #     x,y=minibatch
    #     print( x.cpu().size(), '| batch y: ', y.cpu().size())

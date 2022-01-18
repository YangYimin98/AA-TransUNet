"""cloud_cover_dataset.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: dataloader for cloud cover dataset
"""
import os
from torch.utils.data import Dataset
import numpy as np


class cloud_maps(Dataset):
    def __init__(self, folder, train=True, input_imgs=4, output_imgs=6):
        super(cloud_maps, self).__init__()
        self.train = train

        self.folder_name = os.path.join(folder,'train' if self.train else 'test')
        self.input_imgs = input_imgs
        self.output_imgs = output_imgs

        # Dataset is all the images
        self.dataset = os.listdir(self.folder_name)

        self.size_dataset = len(self.dataset)

    def __getitem__(self, index):

        imgs = np.load(os.path.join(self.folder_name, self.dataset[index]))['arr_0']

        input_img = np.transpose(imgs[:, :, :self.input_imgs], axes=[2, 0, 1]).astype(dtype="float32")
        target_imgs = np.transpose(imgs[:, :, -self.output_imgs:], axes=[2, 0, 1]).astype(dtype="float32")

        return input_img, target_imgs

    def __len__(self):
        return self.size_dataset
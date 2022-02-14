"""precipitation_dataset.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: dataloader for precipatation maps dataset
Ref: https://github.com/HansBambel/SmaAt-UNet

"""

from torch.utils.data import Dataset
import h5py
import numpy as np


class precipitation_maps_oversampled_h5(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super(precipitation_maps_oversampled_h5, self).__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, 'r')["train" if train else "test"]['images'].shape
        self.num_input = num_input_images
        self.num_output = num_output_images
        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, 'r', rdcc_nbytes=1024**3)["train" if self.train else "test"]['images']
        imgs = np.array(self.dataset[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[:self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.samples

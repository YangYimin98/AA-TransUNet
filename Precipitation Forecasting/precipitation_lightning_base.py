"""precipitation_lightning_base.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: precipitation maps dataloader for training sets, test sets and validation sets
"""
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np
from Precipitation_Forecasting.precipitation_dataset import precipitation_maps_oversampled_h5
from Precipitation_Forecasting.precipitation_lightning import AA_TransUnet_base


class Precip_regression_base(TransUnet_base):
    def __init__(self, hparams):
        super(Precip_regression_base, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        train_transform = None
        valid_transform = None
        if self.hparams['use_oversampled_dataset']:
            self.train_dataset = precipitation_maps_oversampled_h5(
                in_file=self.hparams['dataset_folder'], num_input_images=self.hparams['num_input_images'],
                num_output_images=self.hparams['num_output_images'], train=True,
                transform=train_transform
            )
            self.valid_dataset = precipitation_maps_oversampled_h5(
                in_file=self.hparams['dataset_folder'], num_input_images=self.hparams['num_input_images'],
                num_output_images=self.hparams['num_output_images'], train=True,
                transform=valid_transform
            )

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.hparams['valid_size'] * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams['batch_size'], sampler=self.train_sampler,
            num_workers=2, pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.hparams['batch_size'], sampler=self.valid_sampler,
            num_workers=2, pin_memory=True
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams['batch_size'], sampler=self.test_sampler,
            num_workers=2, pin_memory=True
        )
        return test_loader
"""cloud_cover_lightning_base.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: cloud cover dataloader for training sets, test sets and validation sets
Ref: https://github.com/HansBambel/SmaAt-UNet

"""

class Cloud_base(TransUnet_base):

    def __init__(self, hparams):
        super(Cloud_base, self).__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    def prepare_data(self):
        self.train_dataset = cloud_maps(
            folder=self.hparams['dataset_folder'], train=True, input_imgs=self.hparams['num_input_images'],
            output_imgs=self.hparams['num_output_images']
        )
        self.valid_dataset = cloud_maps(
            folder=self.hparams['dataset_folder'], train=True, input_imgs=self.hparams['num_input_images'],
            output_imgs=self.hparams['num_output_images']
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
            num_workers=4, pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.hparams['batch_size'], sampler=self.valid_sampler,
            num_workers=4, pin_memory=True
        )
        return valid_loader
    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams['batch_size'], sampler=self.test_sampler,
            num_workers=2, pin_memory=True
        )
        return test_loader

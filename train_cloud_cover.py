"""train_precipitation"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Run this to train the model on cloud cover dataset
"""

import torch
from Models.AA_TransUNet import AA_TransUnet
import pytorch_lightning as pl

if __name__ == "__main__":
    args = {

        'vit_blocks': 1,
        'vit_heads': 1,
        'vit_dim_linear_mhsa_block':3072,
        'patch_size': 2,
        'vit_transformer_dim': 1024,
        'vit_transformer': None,
        'vit_channels': None,
        'classes': 6,
        'img_dim': 256,
        "in_channels": 4,
        "batch_size": 6,
        "learning_rate": 0.001,
        'gpus': -1,
        "lr_patience": 4,
        "es_patience": 30,
        "use_oversampled_dataset": True,
        "bilinear": True,
        "num_input_images": 4,
        "num_output_images": 6,
        "valid_size": 0.1,
        "dataset_folder": "/AA_TransUNet/Data_cloud_cover_preprocessed"
}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = AA_TransUnet(hparams=args)
    net = net.to(device)
    trainer = pl.Trainer(gpus=-1,
                         fast_dev_run=False,
                         weights_summary='top',
                         max_epochs=100)
    trainer.fit(net)
    trainer.save_checkpoint('AA_TransUNet/results/Model_Saved/1.ckpt')


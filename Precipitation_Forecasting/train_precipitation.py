"""train_precipitation"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Run this to train the model
"""

import torch
from Models.AA_TransUNet import AA_TransUnet
import pytorch_lightning as pl

if __name__ == "__main__":
    args = {

        'vit_blocks': 1,  # MLP layers
        'vit_heads': 1,  # MSA layers
        'vit_dim_linear_mhsa_block': 3072,
        'patch_size': 2,
        'vit_transformer_dim': 768,
        'vit_transformer': None,
        'vit_channels': None,
        'classes': 1,  # classes of outputs.
        'img_dim': 288,  # size of images.
        "in_channels": 12,
        "batch_size": 6,
        "learning_rate": 0.001,
        'gpus': -1,
        "lr_patience": 4,  # learning rate decay.
        "es_patience": 30,  # If the MSE value of the validation set does not decrease in 30 epochs, stop training.
        "use_oversampled_dataset": True,  # Use the smaller dataset for training.
        "bilinear": True,  # if bilinear, use the normal convolutions to reduce the number of channels
        "num_input_images": 12,  # length of input image sequence.
        "num_output_images": 6,  # length of output image sequence.
        "valid_size": 0.1,  # size of validation dataset.
        "dataset_folder": "AA_TransUNet/dataset/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5"
}  # We use NL-50 for training.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = AA_TransUnet(hparams=args)
    net = net.to(device)
    trainer = pl.Trainer(gpus=-1,
                         fast_dev_run=False,
                         weights_summary='top',
                         max_epochs=200)
    trainer.fit(net)
    trainer.save_checkpoint('AA_TransUNet/results/Model_Saved/1.ckpt')


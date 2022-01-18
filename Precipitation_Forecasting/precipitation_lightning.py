"""precipitation_lightning.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Essential TransUnet_base for using pytorch_lightning
"""
from torch import nn, optim
import torch.nn.functional
import pytorch_lightning as pl


class TransUnet_base(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                              mode="min",
                                                              factor=0.1,
                                                              patience=self.hparams['lr_patience']),
            'monitor': 'val_loss',  # Default: val_loss
        }
        return [opt], [scheduler]

    def loss_func(self, y_pred, y_true):
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)

    def training_step(self, batch, batch_idx):
        x, y = batch[0].to('cuda'), batch[1].to('cuda')
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = 0.0
        for output in outputs:
            loss_mean += output['loss']

        loss_mean /= len(outputs)
        return {"log": {"train_loss": loss_mean},
                "progress_bar": {"train_loss": loss_mean}}

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].to('cuda'), batch[1].to('cuda')
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["val_loss"]
        avg_loss /= len(outputs)
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs,
                "progress_bar": {"val_loss": avg_loss}}

    def test_step(self, batch, batch_idx):
        x, y = batch[0].to('cuda'), batch[1].to('cuda')
        y_pred = self(x)
        val_loss = self.loss_func(y_pred.squeeze(), y)
        return {"test_loss": val_loss}

    def test_epoch_end(self, outputs):
        avg_loss = 0.0
        for output in outputs:
            avg_loss += output["test_loss"]
        avg_loss /= len(outputs)
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs,
                "progress_bar": {"test_loss": avg_loss}}
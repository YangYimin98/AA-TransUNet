"""evaluate_precipitation.py"""

"""
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Run this to evaluate the trained model
Ref: https://github.com/HansBambel/SmaAt-UNet
"""

import torch
from torch import nn
import numpy as np
import os
import pickle
from tqdm import tqdm
from Models.AA_TransUNet import AA_TransUnet
from Precipitation_Forecasting.precipitation_dataset import precipitation_maps_oversampled_h5


def compute_loss(model, test_dl, loss="mse", denormalize=True):
    model.eval()  # or model.freeze()?
    model.to("cuda")
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    factor = 1
    if denormalize:
        factor = 47.83
    # go through test set

    with torch.no_grad():

        threshold = 0.5
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to("cuda")
            y_true = y_true.to("cuda")
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze() * factor, y_true.squeeze() * factor,
                                    reduction='sum') / y_true.size(0)

            y_pred_adj = y_pred.squeeze() * 47.83 * 12
            y_true_adj = y_true.squeeze() * 47.83 * 12
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold

            tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            # get metrics for sample
            precision = total_tp / (total_tp + total_fp)
            recall = total_tp / (total_tp + total_fn)
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            f1 = 2 * precision * recall / (precision + recall)
            csi = total_tp / (total_tp + total_fn + total_fp)
            far = total_fp / (total_tp + total_fp)
            hss = (total_tp * total_tn - total_fp * total_fn) / (
                        (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))

        loss_model /= len(test_dl)
        loss_model /= 82944

    return np.array(loss_model.cpu()), precision, recall, accuracy, f1, csi, far, hss


def evaluate(model_folder, data_file, loss, denormalize):
    test_losses = dict()
    dataset = precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # load the models

    model_name = 'AA_TransUNet'
    model = AA_TransUnet.load_from_checkpoint(
        '/AA_TransUNet/results/Model_Saved/TransUnet_Model_Saved_100_Epochs_Final_21.ckpt')
    model_loss, precision, recall, accuracy, f1, csi, far, hss = get_model_loss(model, test_dl, loss,
                                                                                denormalize=denormalize)
    test_losses[model_name] = model_loss
    print(
        f"Model Name: {model_name}, Loss(MSE): {model_loss}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}, hss: {hss}")
    return test_losses


def get_persistence_metrics(test_dl, loss="mse", denormalize=True):
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss

    factor = 1
    if denormalize:
        factor = 47.83
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model = 0.0

    for x, y_true in tqdm(test_dl, leave=False):
        y_pred = x[:, -1, :, :]
        loss_model += loss_func(y_true.squeeze() * factor, y_pred.squeeze() * factor, reduction="mean") / y_true.size(0)
        # denormalize and convert from mm/5min to mm/h
        y_pred_adj = y_pred.squeeze() * 47.83 * 12
        y_true_adj = y_true.squeeze() * 47.83 * 12
        # convert to masks for comparison
        y_pred_mask = y_pred_adj > threshold
        y_true_mask = y_true_adj > threshold

        tn, fp, fn, tp = np.bincount(y_true_mask.view(-1) * 2 + y_pred_mask.view(-1), minlength=4)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        # get metrics for sample
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = (total_tp * total_tn - total_fp * total_fn) / (
                    (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))
    loss_model /= len(test_dl)
    # loss_model /= 82944
    return loss_model, precision, recall, accuracy, f1, csi, far, hss


def print_persistent_metrics(data_file):
    dataset = precipitation_maps_oversampled_h5(
        in_file=data_file,
        num_input_images=12,
        num_output_images=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    loss_model, precision, recall, accuracy, f1, csi, far, hss = get_persistence_metrics(test_dl, loss="mse",
                                                                                         denormalize=True)
    print(
        f"Loss Persistence (MSE): {loss_model}, precision: {precision}, recall: {recall}, accuracy: {accuracy}, f1: {f1}, csi: {csi}, far: {far}, hss: {hss}")
    return loss_model


if __name__ == '__main__':
    loss = "mse"
    denormalize = True
    model_folder = 'AA_TransUNet'
    data_file = 'AA_TransUNet/dataset/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_20.h5'

    load = False  # This changes whether to load or to run the model loss calculation

    # print_persistent_metrics(data_file)
    if load:
        # load the losses
        with open(
                model_folder + f"/results/Metrics_Saved/model_losses_{loss.upper()}_denormalized_1.pkl",
                "rb") as f:
            test_losses = pickle.load(f)
            print(test_losses)
    else:
        test_losses = get_model_losses(model_folder, data_file, loss, denormalize)
        # Save losses
        with open(
                model_folder + f"/results/Metrics_Saved/model_losses_{loss.upper()}_{f'de' if denormalize else ''}_normalized_1.pkl",
                "wb") as f:
            pickle.dump(test_losses, f)

import torch
from torch import nn
import numpy as np
import os
import pickle
from tqdm import tqdm


def compute_loss(model, test_dl, loss="mse"):
    model.eval()
    model.to("cuda")
    if loss.lower() == "mse":
        loss_func = nn.functional.mse_loss
    elif loss.lower() == "mae":
        loss_func = nn.functional.l1_loss
    elif loss.lower() == "bcewl":
        loss_func = nn.functional.binary_cross_entropy_with_logits

    with torch.no_grad():
        loss_model = 0.0
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to("cuda")
            y_true = y_true.to("cuda")
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze(), y_true)
        loss_model /= len(test_dl)
    return np.array(loss_model.cpu())


def evaluate(data_file, model_folder, loss):
    test_losses = dict()
    dataset = cloud_maps(
        folder=data_file,
        input_imgs=4,
        output_imgs=6, train=False)

    test_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # load the models

    model_name = 'TransUnet'
    model = AA_TransUnet.load_from_checkpoint(
        '/AA_TransUNet/results/Model_Saved/T21_CBAM_end_100.ckpt')
    model_loss = get_model_loss(model, test_dl, loss)
    test_losses[model_name] = model_loss
    print(
        f"Model Name: {model_name}, Loss(MSE): {model_loss}")
    return test_losses


if __name__ == '__main__':
    loss = "mse"
    denormalize = True
    model_folder = '/AA_TransUNet/dataset/Data_cloud_cover_preprocessed'
    data_file = "AA_TransUNet/dataset/Data_cloud_cover_preprocessed"

    load = False
    if load:
        with open(model_folder + f"/results/Metrics_Saved/model_losses_{loss.upper()}_denormalized_11_26_50_TransUnet.pkl", "rb") as f:
            test_losses = pickle.load(f)
            print(test_losses)
    else:
        test_losses = get_model_losses(model_folder, data_file, loss)
        with open(
                model_folder + f"/results/Metrics_Saved/model_losses_{loss.upper()}_{f'de' if denormalize else ''}_normalized_1.pkl",
                "wb") as f:
            pickle.dump(test_losses, f)
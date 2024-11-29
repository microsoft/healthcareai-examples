# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import os
import time

from .data import FeatureDataset
from .device import get_device


def create_data_loader(samples, csv, mode, batch_size, num_workers=2, pin_memory=True):
    """
    Creates a data loader for the generated embeddings.

    Args:
    - samples (dict): Dictionary containing the features and image names.
    - csv (pandas.DataFrame): DataFrame containing the labels.
    - mode (str): Mode of the data loader (train or test).
    - batch_size (int): Batch size for the data loader.
    - num_workers (int): Number of workers for the data loader (default: 2).
    - pin_memory (bool): Whether to pin the memory for the data loader (default: True).

    Returns:
    - data_loader (torch.utils.data.DataLoader): Data loader for the generated embeddings.
    """
    ds = FeatureDataset(samples, csv=csv, mode=mode)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


def trainer(train_ds, test_ds, model, loss_function_ts, optimizer, epochs, root_dir):
    """
    Trains a classification model and evaluates it on a validation set.
    Saves the model with the best validation ROC AUC score.
    """

    start_time = time.time()

    max_epoch = epochs
    best_metric = -1
    best_acc = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    # Set device
    device = get_device()
    model = model.to(device)

    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{max_epoch}")
        model.train()
        epoch_loss = 0
        step = 0

        # Training loop
        for batch_idx, (features, pathology_label, img_name) in tqdm(
            enumerate(train_ds),
            total=len(train_ds),
            desc=f"Train Epoch={epoch}",
            ncols=80,
            leave=False,
        ):

            step += 1
            features = features.to(device)
            pathology_label = pathology_label.to(device)

            optimizer.zero_grad()
            _, pred_pathology = model(features)

            loss = loss_function_ts(pred_pathology, pathology_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print(f"{step}/{len(train_ds)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            y_pred_list = []
            y_true_list = []

            for batch_idx, (features, pathology_label, img_name) in tqdm(
                enumerate(test_ds),
                total=len(test_ds),
                desc=f"Test Epoch={epoch}",
                ncols=80,
                leave=False,
            ):

                features = features.to(device)
                pathology_label = pathology_label.to(device)

                _, pred_pathology = model(features)

                y_pred_list.append(pred_pathology)
                y_true_list.append(pathology_label)

            # Concatenate predictions and true labels
            y_pred = torch.cat(y_pred_list, dim=0)
            y_true = torch.cat(y_true_list, dim=0)

            # Compute probabilities for the positive class
            y_scores = torch.softmax(y_pred, dim=1).cpu().numpy()
            y_true_np = y_true.cpu().numpy()

            # Compute ROC AUC
            auc = roc_auc_score(y_true_np, y_scores, multi_class="ovr")

            # Compute accuracy
            acc_metric = (y_pred.argmax(dim=1) == y_true).sum().item() / len(y_true)

            metric_values.append(auc)

            # Save the best model
            if auc > best_metric:
                best_metric = auc
                best_acc = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print("Saved new best metric model")

            print(
                f"Current epoch: {epoch + 1} Current AUC: {auc:.4f}"
                f" Current accuracy: {acc_metric:.4f}"
                f" Best AUC: {best_metric:.4f}"
                f" Best accuracy: {best_acc:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    end_time = time.time()
    training_time = end_time - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Training Time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )
    return best_acc, best_metric

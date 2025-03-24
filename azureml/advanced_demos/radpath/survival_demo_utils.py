import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch import nn
from torch.nn import Parameter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from tqdm import tqdm
import time
import warnings

from healthcareai_toolkit.data.io import normalize_image_to_uint8, read_nifti


# Training
## Adapter Network Model Architecture
class SurvivalModel(nn.Module):
    def __init__(self, in_channels_rad, in_channels_path, hidden_dim, num_class):
        super().__init__()

        self.in_channels_rad = int(in_channels_rad)
        self.in_channels_path = int(in_channels_path)
        self.hidden_dim = int(hidden_dim)
        self.num_class = num_class

        ## Fuse multi-channel MRI Radiology Embeddings
        self.fuse_rad = nn.Sequential(
            nn.Linear(self.in_channels_rad * 4, self.in_channels_rad),
            nn.GELU(),
            nn.Linear(self.in_channels_rad, self.in_channels_rad),
            nn.LayerNorm(self.in_channels_rad),
        )

        self.pathology_pool = nn.AdaptiveAvgPool2d((1, 1))

        ## Adaptor Module
        self.rad_embd = nn.Sequential(
            nn.Linear(self.in_channels_rad, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.path_embd = nn.Sequential(
            nn.Linear(self.in_channels_path, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.retrieval_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=512 * 2,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
        )

        ## Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.num_class)
        )  ## num_class: 1
        self.act = nn.Sigmoid()

        ## Set range for hazard value
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, t1, t1_pc, t2, flair, path):
        ## Fuse Different Echo Sequence MRI Radiology Embeddings
        radiology_feat = torch.cat(
            [t1.float(), t1_pc.float(), t2.float(), flair.float()], dim=1
        )  ## 4x1024
        radiology_feat = self.fuse_rad(
            radiology_feat.view(radiology_feat.size(0), -1)
        )  ## Convert to 1-D vector
        radiology_feat = self.rad_embd(radiology_feat)  ## 1x1024 -> 1x512

        ## Pool Pathology Embeddings
        pathology_feat = (
            self.pathology_pool(path).squeeze(4).squeeze(3)
        )  ## 1x1536x14x14 -> 1x1536x1x1 -> 1x1536
        pathology_feat = self.path_embd(pathology_feat)  ## 1x1536x1x1 -> 1x512

        ## Concatenate Radiology and Pathology Embeddings
        feat = torch.cat([radiology_feat, pathology_feat.squeeze(1)], dim=1)  ## 1x1024

        ## Fuse Multi-Modality Embeddings with 2-Layer Convolution
        feat = self.retrieval_conv(torch.unsqueeze(feat, 2))

        ## Compute Hazard Value
        hazard = self.prediction_head(feat.squeeze(2))
        hazard = self.act(hazard)
        hazard = self.output_range * hazard + self.output_shift

        return hazard


## Survival Data Loader
class SurvivalDataset(data.Dataset):
    def __init__(self, csv_file, radiology_embeddings, pathology_embeddings):
        self.csv_data = pd.read_csv(csv_file)

        ## Load radiology embeddings folder path
        self.radiology_embeddings = radiology_embeddings

        ## Load pathology embeddings
        self.pathology_embeddings = pathology_embeddings

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        subj_id = self.csv_data.iloc[idx]["TCGA ID"]
        grade = self.csv_data.iloc[idx]["Grade"]
        censor = self.csv_data.iloc[idx]["censored"]
        survival_time = (
            self.csv_data.iloc[idx]["Survival dates"] / 365
        )  ## Convert to years

        ## Load Radiology Embeddings
        t1_embedding = self.radiology_embeddings[subj_id]["0000"]  ## T1 (1x1024)
        t1pc_embedding = self.radiology_embeddings[subj_id][
            "0001"
        ]  ## T1 Post-Contrast (1x1024)
        t2_embedding = self.radiology_embeddings[subj_id]["0002"]  ## T2 (1x1024)
        t2flair_embedding = self.radiology_embeddings[subj_id][
            "0003"
        ]  ## T2 FLAIR (1x1024)

        ## Load Pathology Embeddings
        path_embedding = self.pathology_embeddings[subj_id]  ## Pathology (1x1536x14x14)

        return {
            "subj_id": subj_id,
            "grade": grade,
            "censor": censor,
            "survival_time": survival_time,
            "t1": t1_embedding,
            "t1pc": t1pc_embedding,
            "t2": t2_embedding,
            "flair": t2flair_embedding,
            "path": path_embedding,
        }


def create_survival_data_loader(
    csv,
    radiology_embeddings,
    pathology_embeddings,
    mode,
    batch_size,
    num_workers=2,
    pin_memory=True,
):
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
    ds = SurvivalDataset(csv, radiology_embeddings, pathology_embeddings)
    if mode == "train":
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif mode == "test":
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return data_loader


## Survival Model Trainer Function
def survival_trainer(train_ds, test_ds, model, optimizer, epochs, root_dir):
    """
    Trains a survival prediction adaptor model and evaluates it on a validation set.
    Saves the model with the best C-Index score.

    Args:
        train_ds (Dataset): Training dataset.
        test_ds (Dataset): Testing dataset.
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        root_dir (str): Directory to save the best model.

    Returns:
        dict: Dictionary containing the best C-index, p-value, and censored prediction accuracy on the test set.
    """

    start_time = time.time()  # Record the start time of training

    max_epoch = epochs  # Maximum number of epochs
    best_cindex = 0  # Initialize the best C-index

    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move the model to the device

    for epoch in range(max_epoch):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{max_epoch}")
        model.train()  # Set the model to training mode
        epoch_loss = 0  # Initialize epoch loss
        step = 0  # Initialize step counter

        # Training loop
        for batch_idx, input_data in tqdm(
            enumerate(train_ds),
            total=len(train_ds),
            desc=f"Train Epoch={epoch}",
            ncols=80,
            leave=False,
        ):
            step += 1
            # Move input data to the device
            t1, t1_pc, t2, flair = (
                input_data["t1"].to(device),
                input_data["t1pc"].to(device),
                input_data["t2"].to(device),
                input_data["flair"].to(device),
            )
            path = input_data["path"].to(device)
            censor, surv_time = input_data["censor"].to(device), input_data[
                "survival_time"
            ].to(device)

            optimizer.zero_grad()  # Zero the gradients
            pred_hazard = model(t1, t1_pc, t2, flair, path)  # Forward pass

            loss = CoxLoss(surv_time, censor, pred_hazard, device)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            epoch_loss += loss.item()  # Accumulate loss

            print(f"{step}/{len(train_ds)}, train_loss: {loss.item():.4f}")

        epoch_loss /= step  # Compute average loss for the epoch
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        hazard_pred_all, censor_all, survtime_all = (
            np.array([]),
            np.array([]),
            np.array([]),
        )
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, input_data in tqdm(
                enumerate(test_ds),
                total=len(test_ds),
                desc=f"Test Epoch={epoch}",
                ncols=80,
                leave=False,
            ):
                # Move input data to the device
                t1, t1_pc, t2, flair = (
                    input_data["t1"].to(device),
                    input_data["t1pc"].to(device),
                    input_data["t2"].to(device),
                    input_data["flair"].to(device),
                )
                path = input_data["path"].to(device)
                censor, surv_time = input_data["censor"].to(device), input_data[
                    "survival_time"
                ].to(device)

                pred_hazard = model(t1, t1_pc, t2, flair, path)  # Forward pass

                test_loss = CoxLoss(
                    surv_time, censor, pred_hazard, device
                )  # Compute loss
                # Accumulate predictions and labels
                hazard_pred_all = np.concatenate(
                    (hazard_pred_all, pred_hazard.detach().cpu().numpy().reshape(-1))
                )
                censor_all = np.concatenate(
                    (censor_all, censor.detach().cpu().numpy().reshape(-1))
                )
                survtime_all = np.concatenate(
                    (survtime_all, surv_time.detach().cpu().numpy().reshape(-1))
                )

            # Compute evaluation metrics
            cindex_test = CIndex_lifeline(hazard_pred_all, censor_all, survtime_all)
            pvalue_test = cox_log_rank(hazard_pred_all, censor_all, survtime_all)
            surv_acc_test = accuracy_cox(hazard_pred_all, censor_all)

            # Save the best model based on C-index
            if cindex_test >= best_cindex:
                best_cindex = cindex_test
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print("Saved new best metric model")

            print(
                f"Current epoch: {epoch + 1}"
                f" Current C-Index: {cindex_test:.4f}"
                f" Best C-Index: {best_cindex:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    end_time = time.time()  # Record the end time of training
    training_time = end_time - start_time  # Compute total training time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total Training Time: {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
    print(
        f"Training completed, Best C-Index: {best_cindex:.4f} at epoch: {best_metric_epoch}"
    )
    return {
        "best_cindex": best_cindex,
        "p_value": pvalue_test,
        "survival_acc_test": surv_acc_test,
    }


## ------------------------------------------------------------------------------------------------- ##


## Loss function and evaluation metric
def CoxLoss(survtime, censor, hazard_pred, device):
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    temp = exp_theta * R_mat
    loss_cox = -torch.mean(
        (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
    )
    return loss_cox


def CIndex_lifeline(hazards, labels, survtime_all):
    """
    Calculate the concordance index (C-index) for survival predictions.

    The C-index is a measure of the predictive accuracy of a survival model. It quantifies the
    concordance between the predicted risk scores and the actual survival times.

    Parameters:
    hazards (np.array): Array of predicted hazard values.
    labels (np.array): Array of event/censoring labels (1 if event occurred, 0 if censored).
    survtime_all (np.array): Array of survival times.

    Returns:
    float: The concordance index (C-index).
    """
    # Calculate the concordance index using the lifelines library's concordance_index function.
    # The concordance_index function takes the survival times, predicted risk scores (negative hazards),
    # and event/censoring labels as inputs and returns the C-index.
    c_index = concordance_index(survtime_all, -hazards, labels)

    return c_index


def cox_log_rank(hazardsdata, labels, survtime_all):
    """
    Perform the log-rank test to compare the survival distributions of two groups.

    Parameters:
    hazardsdata (np.array): Array of predicted hazard values.
    labels (np.array): Array of event/censoring labels (1 if event occurred, 0 if censored).
    survtime_all (np.array): Array of survival times.

    Returns:
    float: p-value from the log-rank test.
    """
    # Calculate the median of the hazard data to dichotomize the data into two groups
    median = np.median(hazardsdata)

    # Initialize an array to store the dichotomized hazard data
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)

    # Assign 1 to the elements greater than the median, 0 otherwise
    hazards_dichotomize[hazardsdata > median] = 1

    # Create a boolean index for the first group (hazards <= median)
    idx = hazards_dichotomize == 0

    # Split the survival times and labels into two groups based on the dichotomized hazard data
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]

    # Perform the log-rank test between the two groups
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)

    # Extract the p-value from the test results
    pvalue_pred = results.p_value

    return pvalue_pred


def accuracy_cox(hazardsdata, labels):
    """
    Calculate the accuracy of the Cox model predictions.

    This function computes the accuracy of the Cox model by comparing the predicted survival events
    (based on the median of the predicted hazard values) against the true survival events.

    Parameters:
    hazardsdata (np.array): Array of predicted hazard values.
    labels (np.array): Array of event/censoring labels (1 if event occurred, 0 if censored).

    Returns:
    float: The accuracy of the Cox model predictions.
    """
    # Calculate the median of the predicted hazard values
    median = np.median(hazardsdata)

    # Initialize an array to store the dichotomized hazard data
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)

    # Assign 1 to the elements greater than the median, 0 otherwise
    hazards_dichotomize[hazardsdata > median] = 1

    # Calculate the number of correct predictions by comparing the dichotomized hazard data with the true labels
    correct = np.sum(hazards_dichotomize == labels)

    # Calculate the accuracy as the ratio of correct predictions to the total number of labels
    accuracy = correct / len(labels)

    return accuracy


## ------------------------------------------------------------------------------------------------- ##


## Perform Inference
def perform_survival_inference(model, test_loader):
    """
    Perform inference on the test dataset using the trained model.

    Args:
        model (nn.Module): The trained model.
        test_loader (Dataset): The test dataset.

    Returns:
        tuple: A tuple containing:
            - quant_result (dict): Dictionary with C-index, p-value, and survival accuracy.
            - prediction (dict): Dictionary with subject IDs, predicted hazards, censoring status, and survival times.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode

    # Initialize arrays to store predictions and labels
    hazard_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    subject_id_all = []  # List to store subject IDs

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, input_data in tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc="Inference",
            ncols=80,
            leave=False,
        ):
            # Move input data to the device
            t1, t1_pc, t2, flair = (
                input_data["t1"].to(device),
                input_data["t1pc"].to(device),
                input_data["t2"].to(device),
                input_data["flair"].to(device),
            )
            path = input_data["path"].to(device)
            censor, surv_time = input_data["censor"].to(device), input_data[
                "survival_time"
            ].to(device)

            # Forward pass to get predicted hazards
            pred_hazard = model(t1, t1_pc, t2, flair, path)

            # Accumulate predictions and labels
            hazard_pred_all = np.concatenate(
                (hazard_pred_all, pred_hazard.detach().cpu().numpy().reshape(-1))
            )
            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )
            survtime_all = np.concatenate(
                (survtime_all, surv_time.detach().cpu().numpy().reshape(-1))
            )

            for item in input_data["subj_id"]:
                subject_id_all.append(item)

        # Compute evaluation metrics
        cindex_test = CIndex_lifeline(hazard_pred_all, censor_all, survtime_all)
        pvalue_test = cox_log_rank(hazard_pred_all, censor_all, survtime_all)
        surv_acc_test = accuracy_cox(hazard_pred_all, censor_all)

        # Store evaluation metrics in a dictionary
        quant_result = {
            "cindex": cindex_test,
            "p_value": pvalue_test,
            "survival_acc": surv_acc_test,
        }

        # Store predictions in a dictionary
        prediction = {
            "subject_id": subject_id_all,
            "hazard_pred": hazard_pred_all,
            "censored": list(censor_all),
            "survival_time": list(survtime_all),
        }

    return quant_result, prediction


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_trained_model(model, model_path):
    # Load Model State
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"You are using `torch.load` with `weights_only=False`.*",
    )
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model


## ------------------------------------------------------------------------------------------------- ##


## Survival Prediction Demo Utility Functions
def read_nifti_client(file_path):
    image_data = read_nifti(file_path)
    image_data = normalize_image_to_uint8(image_data, percentiles=(1, 99))
    return image_data.astype(np.uint8)


def filter_data(csv_file, radiology_embeddings_folder, label_key=None):
    """
    Filters the data based on the presence of radiology and pathology images.

    Parameters:
    csv_file (str): Path to the CSV file containing the data.
    radiology_embeddings_folder (str): Path to the folder containing radiology images.

    Returns:
    df_filtered (pd.DataFrame): Filtered DataFrame containing the data.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Filter the DataFrame based on the label key if provided
    if label_key is not None:
        subj_list = [
            subj.split("_")[0]
            for subj in os.listdir(radiology_embeddings_folder)
            if "_" in subj
        ]
        df_filtered = df[df["TCGA ID"].isin(subj_list)]
        df_filtered = df_filtered[label_key]
    else:
        df_filtered = df

    return df_filtered


def hazard2grade(hazard, p):
    """
    Convert a hazard value to a grade based on provided thresholds.

    This function takes a hazard value and a list of thresholds, and returns
    the grade corresponding to the first threshold that the hazard value is
    less than. If the hazard value is greater than or equal to all thresholds,
    the function returns the length of the threshold list.

    Args:
        hazard (float): The hazard value to be converted to a grade.
        p (list of float): A list of threshold values in ascending order.

    Returns:
        int: The grade corresponding to the hazard value.
    """
    for i in range(len(p)):
        if hazard < p[i]:
            return i
    return len(p)


def plot_grade_distribution_correspondence(csv_file, radiology_folder):
    """
    Plots the grade distribution and the correspondence between survival months and grade.

    Parameters:
    csv_file (str): Path to the CSV file containing the data.
    radiology_folder (str): Path to the folder containing radiology images.

    Returns:
    None
    """
    # Filter the data based on the presence of radiology images and the categorical information that we want to extract
    df_filtered = filter_data(
        csv_file,
        radiology_folder,
        label_key=["TCGA ID", "Grade", "Survival dates", "censored"],
    )
    df_filtered["Survival dates"] = df_filtered["Survival dates"] / 365

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Plot the grade distribution
    df_filtered["Grade"] = df_filtered["Grade"].astype(int)
    grade_counts = df_filtered["Grade"].value_counts().sort_index()
    axes[0].bar(
        grade_counts.index, grade_counts.values, color="skyblue", edgecolor="black"
    )
    axes[0].set_xlabel("Tumor Grading")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_ylabel("Number of Subjects")
    axes[0].set_title("Grade Distribution of Subjects")

    # Plot the correspondence between survival dates and grade
    df_filtered.boxplot(
        column="Survival dates",
        by="Grade",
        grid=False,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="skyblue", color="black"),
        medianprops=dict(color="red"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(markerfacecolor="red", marker="o", markersize=5),
        ax=axes[1],
    )
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Survival Times (Years)")
    axes[1].set_title("Survival Time (Years) by Tumor Grading")

    # Suppress the default title to avoid overlap
    plt.suptitle("")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()


def visualize_radiology_pathology_images(
    csv_file, radiology_image_folder, pathology_image_folder
):
    # Filter the data based on the presence of radiology images and the categorical information that we want to extract
    df_filtered = filter_data(
        csv_file,
        radiology_image_folder,
        label_key=["TCGA ID", "Grade"],
    )

    # Sample one subject from each grade
    sampled_subjects = (
        df_filtered.groupby("Grade").apply(lambda x: x.sample(1)).reset_index(drop=True)
    )

    # Iterate through the sampled subjects and visualize their diagnostic slices and pathology patches
    image_modality = {
        0: "T1 Weighted",
        1: "T1 Post-Contrast",
        2: "T2 Weighted",
        3: "T2 FLAIR",
        4: "Pathology",
    }
    for idx, row in sampled_subjects.iterrows():
        random_subject = row["TCGA ID"]
        grade = int(row["Grade"])

        # Get the corresponding diagnostic slices and pathology patches
        image_list = [
            os.path.join(radiology_image_folder, f"{random_subject}_0000.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0001.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0002.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0003.png"),
            os.path.join(pathology_image_folder, f"{random_subject}.png"),
        ]

        # Create a figure to display all diagnostic slices
        fig, axes = plt.subplots(1, len(image_list), figsize=(20, 5))
        fig.suptitle(
            f"Survival Prediction Input Image for Grade {grade}: {random_subject}",
            fontsize=16,
        )

        for i, slice_path in enumerate(image_list):
            # Read the NIfTI file
            image_slice = Image.open(slice_path)
            image_slice_array = np.array(image_slice)
            image_slice_array = np.flipud(image_slice_array)

            # Display the slice
            if i == 4:
                axes[i].imshow(image_slice_array)
            else:
                axes[i].imshow(image_slice_array, cmap="gray")
            axes[i].set_title(image_modality[i], fontsize=12)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


def plot_survival_curve(csv_file, radiology_folder=None, split=False):
    """
    Plots the survival curve for the subjects based on the categorical labels survival months.

    Parameters:
    csv_file (str): Path to the CSV file containing the patients' information.
    radiology_folder (str): Path to the folder containing radiology images, it used to filter patients information.
    split (bool, optional): If the csv file input contains training / testing split subjects only, set to True. Default is False.

    Returns:
    None
    """
    # Filter the data based on the presence of radiology images and the categorical information that we want to extract
    if split:
        df_filtered = pd.read_csv(csv_file)
    else:
        df_filtered = filter_data(
            csv_file,
            radiology_folder,
            label_key=["TCGA ID", "Grade", "Survival dates", "censored"],
        )

    # Create a Kaplan-Meier Fitter object
    kmf = KaplanMeierFitter()

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = plt.subplot()
    censor_style = {"ms": 20, "marker": "+"}

    # Plot the Kaplan-Meier survival curves for each grade
    for i, label in enumerate(["Grade 0", "Grade 1", "Grade 2"]):
        temp = df_filtered[df_filtered["Grade"] == i]
        kmf.fit(
            durations=temp["Survival dates"] / 365,
            event_observed=temp["censored"],
            label=label,
        )
        if label == "Grade 0":
            kmf.plot(
                ax=ax,
                show_censors=True,
                ci_show=False,
                c="b",
                linewidth=4,
                ls="-",
                censor_styles=censor_style,
            )
        elif label == "Grade 1":
            kmf.plot(
                ax=ax,
                show_censors=True,
                ci_show=False,
                c="orange",
                linewidth=4,
                ls="-",
                censor_styles=censor_style,
            )
        elif label == "Grade 2":
            kmf.plot(
                ax=ax,
                show_censors=True,
                ci_show=False,
                c="r",
                linewidth=4,
                ls="-",
                censor_styles=censor_style,
            )

    # Set the title and labels for the plot
    plt.title("Kaplan-Meier Survival Curves by Ground Truth Tumor Grade")
    plt.xlabel("Survival Time (Years)")
    plt.ylabel("Survival Probability")

    # Display the plot
    plt.show()


def plot_gt_and_pred_survival_curves(
    csv_file, predictions, radiology_folder=None, is_split=False, percentile=[33, 66]
):
    """
    Plot both ground-truth survival curves (by tumor grade) and predicted-hazard
    survival curves in two subplots.

    Args:
        csv_file (str): Path to the CSV file containing ground-truth data.
        predictions (dict): Dictionary with keys "survival_time", "censored", "hazard_pred".
        radiology_folder (str, optional): Path for filtering data (if needed).
        is_split (bool, optional): Whether to skip filtering. Default is False.
        percentile (list, optional): Percentiles for dichotomizing hazard. Default is [33, 66].
    """

    # Assumes a function filter_data(...) and hazard2grade(...) are already defined.
    # Assumes "Grade", "Survival dates", and "censored" columns exist in the CSV.

    # Prepare ground-truth data
    if is_split:
        df_gt = pd.read_csv(csv_file)
    else:
        df_gt = filter_data(
            csv_file,
            radiology_folder,
            label_key=["TCGA ID", "Grade", "Survival dates", "censored"],
        )

    # Build dataframe from predictions
    df_pred = pd.DataFrame(
        {
            "survival_time": predictions["survival_time"],
            "censored": predictions["censored"],
            "hazard_pred": predictions["hazard_pred"],
        }
    )
    p = np.percentile(df_pred["hazard_pred"], percentile)
    data = []
    for h in df_pred["hazard_pred"]:
        data.append(hazard2grade(h, p))
    df_pred["Grade_pred"] = data

    # Create figure
    fig, axes = plt.subplots(1, len(sorted(df_gt["Grade"].unique())), figsize=(15, 5))
    censor_style = {"ms": 20, "marker": "+"}

    for i, g in enumerate(sorted(df_gt["Grade"].unique())):
        ax_sub = axes[i] if len(sorted(df_gt["Grade"].unique())) > 1 else axes
        grade = int(g)
        # Ground truth
        dtemp_gt = df_gt[df_gt["Grade"] == g]
        kmf_gt = KaplanMeierFitter()
        kmf_gt.fit(
            dtemp_gt["Survival dates"] / 365,
            dtemp_gt["censored"],
            label=f"GT Grade {grade}",
        )
        kmf_gt.plot_survival_function(
            ax=ax_sub,
            ci_show=False,
            censor_styles=censor_style,
            show_censors=True,
            linewidth=4,
        )

        # Predicted
        dtemp_pred = df_pred[df_pred["Grade_pred"] == g]
        kmf_pred = KaplanMeierFitter()
        kmf_pred.fit(
            dtemp_pred["survival_time"],
            dtemp_pred["censored"],
            label=f"Predicted Grade {grade}",
        )
        kmf_pred.plot_survival_function(
            ax=ax_sub,
            ci_show=False,
            censor_styles=censor_style,
            show_censors=True,
            linewidth=4,
        )

        ax_sub.set_title(f"Grade {grade}")
        ax_sub.set_xlabel("Time (Years)")
        ax_sub.set_ylabel("Survival Probability")
        ax_sub.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def visualize_radiology_pathology_images_with_pred_grade(
    csv_file,
    predictions,
    radiology_image_folder,
    pathology_image_folder,
    percentile=[33, 66],
):
    df_filtered = filter_data(
        csv_file,
        radiology_image_folder,
        label_key=["TCGA ID", "Grade"],
    )

    df_pred = pd.DataFrame(
        {
            "subject_id": predictions["subject_id"],
            "survival_time": predictions["survival_time"],
            "censored": predictions["censored"],
            "hazard_pred": predictions["hazard_pred"],
        }
    )

    # Compute predicted grade using hazard2grade
    p = np.percentile(df_pred["hazard_pred"], percentile)
    df_pred["Grade_pred"] = df_pred["hazard_pred"].apply(lambda x: hazard2grade(x, p))

    df_pred = df_pred[df_pred["subject_id"].isin(df_filtered["TCGA ID"])]

    df_pred = df_pred.merge(
        df_filtered[["TCGA ID", "Grade"]],
        left_on="subject_id",
        right_on="TCGA ID",
        how="left",
    )

    radiology_modality = {
        0: "T1 Weighted",
        1: "T1 Post-Contrast",
        2: "T2 Weighted",
        3: "T2 FLAIR",
        4: "Pathology",
    }
    for idx, row in df_pred.iterrows():
        random_subject = row["subject_id"]
        gt_grade = int(row["Grade"])
        pred_grade = int(row["Grade_pred"])

        image_slice_list = [
            os.path.join(radiology_image_folder, f"{random_subject}_0000.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0001.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0002.png"),
            os.path.join(radiology_image_folder, f"{random_subject}_0003.png"),
            os.path.join(pathology_image_folder, f"{random_subject}.png"),
        ]

        fig, axes = plt.subplots(1, len(image_slice_list), figsize=(20, 5))
        fig.suptitle(
            f"Subject {random_subject} - GT Grade: {gt_grade}, Predicted Grade from Hazard: {pred_grade}",
            fontsize=20,
        )

        for i, slice_path in enumerate(image_slice_list):
            image_slice = Image.open(slice_path)
            image_slice_array = np.flipud(np.array(image_slice))
            if i == 4:
                axes[i].imshow(image_slice_array)
            axes[i].imshow(image_slice_array, cmap="gray")
            axes[i].set_title(radiology_modality[i], fontsize=12)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

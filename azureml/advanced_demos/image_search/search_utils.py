import numpy as np
import faiss
import pandas as pd
import torch
from torch.utils import data
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from healthcareai_toolkit.data import io
from PIL import Image


def create_faiss_index(df, feature_column):
    """
    Create a FAISS index for the given feature column in the dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the feature vectors.
    feature_column : str
        Name of the column in the DataFrame containing the feature vectors.

    Returns:
    --------
    faiss.Index
        A FAISS index built from the feature vectors.
    """
    d = df[feature_column].iloc[0].shape[0]  # dimension of the feature vectors
    index = faiss.IndexFlatL2(d)  # create the index
    features = np.stack(df[feature_column].values)  # stack the feature vectors
    index.add(features)  # add vectors to the index
    return index


def evaluate_faiss_search(
    faiss_index, test_df, train_df, k_list, query_column_feature_name="mi2_features"
):
    """
    Evaluate the performance of a FAISS-based nearest neighbor search.

    This function calculates the accuracy of Top-K retrieval for a given test dataset
    and generates three data frames:
    1. summary_df: Overall accuracy for each specified k.
    2. detail_df: Label-wise accuracy with detailed breakdown for each k.
    3. search_results_df: Detailed search results for each query.

    Parameters:
    -----------
    faiss_index : faiss.Index
        A trained FAISS index for nearest neighbor search.
    test_df : pandas.DataFrame
        Test dataset containing:
            - `query_column_feature_name`: Feature vectors stored as NumPy arrays.
            - `Label`: Ground truth labels for evaluation.
            - `Label Category`: Human-readable label categories.
    train_df : pandas.DataFrame
        Training dataset containing:
            - `query_column_feature_name`: Feature vectors stored as NumPy arrays.
            - `Label`: Ground truth labels used to train the FAISS index.
    k_list : list of int
        A list of values of k (e.g., [1, 3, 5]) to evaluate Top-K retrieval accuracy.
    query_column_feature_name : str, optional
        Name of the column in `test_df` and `train_df` containing the feature vectors.
        Default is "mi2_features".

    Returns:
    --------
    summary_df : pandas.DataFrame
        A data frame summarizing overall accuracy at each k.
        Columns:
            - `k (Top-K)`: The value of k.
            - `Overall Accuracy`: The average accuracy across all labels for the given k.
    detail_df : pandas.DataFrame
        A data frame providing label-wise accuracy at each k.
        Columns:
            - `Label`: Ground truth label.
            - `Category`: Human-readable label category.
            - `Accuracy @ k=1`: Accuracy at k=1.
            - `Accuracy @ k=3`: Accuracy at k=3.
            - `Accuracy @ k=5`: Accuracy at k=5, etc., depending on k_list.
    search_results_df : pandas.DataFrame
        A data frame containing detailed search results for each query.
        Columns:
            - `query_name`: Name of the query sample.
            - `query_label`: Ground truth label of the query sample.
            - `retrieved_indices`: Indices of the retrieved samples.
            - `retrieved_labels`: Labels of the retrieved samples.

    Assumptions:
    ------------
    1. The FAISS index is already trained with the features from the training dataset.
    2. Feature vectors in `test_df` and `train_df` are stored as NumPy arrays in the
       specified `query_column_feature_name`.
    3. Labels (`Label`) and categories (`Label Category`) are available for evaluation.

    Example:
    --------
    summary_df, detail_df, search_results_df = evaluate_faiss_search(
        faiss_index, test_df, train_df, k_list=[1, 3, 5]
    )
    """
    # Convert test features to a NumPy array
    test_features = np.stack(test_df[query_column_feature_name].values)
    kmax = max(k_list)

    # Search for the kmax nearest neighbors of the test features
    D, I = faiss_index.search(test_features, kmax)

    # Precompute the labels for the training features
    train_labels = train_df["Label"].values

    # Initialize label accuracy
    label_accuracy = {
        label: {f"precision_at_{k}": [] for k in k_list}
        for label in test_df["Label"].unique()
    }

    # Initialize a list to store search results for each query
    search_results = []

    # Iterate over each test sample
    for i in range(I.shape[0]):
        query_label = test_df.iloc[i]["Label"]
        query_name = test_df.iloc[i]["Name"]
        retrieved_indices = I[i]
        retrieved_labels = train_labels[retrieved_indices]

        # Store the search results for the current query
        search_results.append(
            {
                "query_name": query_name,
                "query_label": query_label,
                "retrieved_indices": retrieved_indices,
                "retrieved_labels": retrieved_labels,
            }
        )

        # Calculate accuracy for each k in k_list
        for k in k_list:
            precision_at_k = np.sum(retrieved_labels[:k] == query_label) / k
            label_accuracy[query_label][f"precision_at_{k}"].append(precision_at_k)

    # Prepare data for summary and detail data frames
    summary_data = []
    detail_data = []

    for k in k_list:
        overall_accuracy_list = []
        for label, accuracies in label_accuracy.items():
            avg_accuracy = np.mean(accuracies[f"precision_at_{k}"])
            label_category = test_df[test_df["Label"] == label]["Label Category"].iloc[
                0
            ]
            detail_data.append(
                {
                    "Label": label,
                    "Category": label_category,
                    f"Precision @ k={k}": avg_accuracy,
                }
            )
            overall_accuracy_list.append(avg_accuracy)

        overall_accuracy = np.mean(overall_accuracy_list)
        summary_data.append({"k (Top-K)": k, "Overall Precision": overall_accuracy})

    # Create data frames
    summary_df = pd.DataFrame(summary_data)
    # Prepare data for summary and detail data frames
    summary_data = []
    detail_data = []

    for label in sorted(test_df["Label"].unique()):
        label_category = test_df[test_df["Label"] == label]["Label Category"].iloc[0]
        label_data = {"Label": label, "Category": label_category}
        for k in k_list:
            avg_accuracy = np.mean(label_accuracy[label][f"precision_at_{k}"])
            label_data[f"Precision @ k={k}"] = avg_accuracy
        detail_data.append(label_data)

    for k in k_list:
        overall_accuracy = np.mean(
            [
                np.mean(label_accuracy[label][f"precision_at_{k}"])
                for label in label_accuracy
            ]
        )
        summary_data.append({"k (Top-K)": k, "Overall Precision": overall_accuracy})

    # Create data frames
    summary_df = pd.DataFrame(summary_data)
    detail_df = pd.DataFrame(detail_data)

    # Convert search results to a DataFrame
    search_results_df = pd.DataFrame(search_results)

    return summary_df, detail_df, search_results_df


def display_query_and_retrieved_images(
    query_df,
    search_results_df,
    cx_image_path,
    train_features_df,
    format=None,
    overlay=False,
    num_labels=5,
    percentiles=(0.1, 0.99),
):
    """
    Visualizes the query images and their top retrieved images based on a FAISS search.

    This function displays the query images in the first column and the top retrieved images in subsequent columns.
    Each row represents a query image with its corresponding retrieved results. The rank of the retrieved images
    is displayed along with their ground truth labels and categories.

    Parameters:
    -----------
    query_df : pandas.DataFrame
        A DataFrame containing the query image samples to be searched. The DataFrame should have at least the following columns:
        - `Name`: The filename of the query image.
        - `Label`: The ground truth label of the query image.
        - `Label Category`: A human-readable label category for the query image.

    search_results_df : pandas.DataFrame
        A DataFrame containing the search results for each query. The DataFrame should include the following columns:
        - `query_name`: The name of the query image.
        - `query_label`: The label of the query image.
        - `retrieved_indices`: Indices of the top retrieved images from the search.
        - `retrieved_labels`: Labels of the top retrieved images from the search.

    cx_image_path : str
        The path to the directory containing the images to be displayed. This is used to construct the full paths for the query and retrieved images.

    train_features_df : pandas.DataFrame
        The DataFrame containing the training dataset used for retrieval. It provides the mapping between the retrieved indices and the corresponding image names.

    num_labels : int, optional (default=5)
        The number of query samples (or rows) to display in the plot. This determines the number of rows in the visualization grid.

    percentiles : tuple, optional (default=(0.1, 0.99))
        The percentiles used for normalizing the image intensity values.

    Returns:
    --------
    None
        This function displays a plot containing the query images and their corresponding retrieved neighbors. It does not return any values.

    Example:
    --------
    display_query_and_retrieved_images(query_df, search_results_df, "/path/to/images", train_features_df, num_labels=5, percentiles=(0.1, 0.99))
    """
    plt.figure(figsize=(20, 5 * num_labels), dpi=300)

    for idx, query_sample in query_df.iterrows():
        # Retrieve the search results for the query sample
        query_results = search_results_df[
            search_results_df["query_name"] == query_sample["Name"]
        ]

        # Display the query image
        if format:
            query_image_path = os.path.join(
                cx_image_path, query_sample["Name"] + "." + format
            )
        else:
            query_image_path = os.path.join(cx_image_path, query_sample["Name"])

        plt.subplot(num_labels, 6, idx * 6 + 1)
        if query_image_path.endswith(".dcm"):
            normalized_image = io.normalize_image_to_uint8(
                io.read_dicom_to_array(query_image_path), percentiles=percentiles
            )
        elif query_image_path.endswith(".png"):
            normalized_image = io.normalize_image_to_uint8(
                np.array(Image.open(query_image_path)), percentiles=percentiles
            )

        ## Check if the image is grayscale or overlaid with segmentation contours
        if overlay:
            plt.imshow(normalized_image)
        else:
            plt.imshow(normalized_image, cmap="gray")

        plt.title(
            f"Query\nLabel: {query_sample['Label']}\nCategory: {query_sample['Label Category']}"
        )
        plt.axis("off")

        # Display the retrieved images with ranking
        retrieved_images = query_results["retrieved_indices"].apply(
            lambda indices: [train_features_df.iloc[i]["Name"] for i in indices]
        )

        for i, (retrieved_image, label) in enumerate(
            zip(retrieved_images.iloc[0], query_results["retrieved_labels"].iloc[0])
        ):
            if format:
                if query_sample["Name"].endswith(".nii.gz"):
                    retrieved_image_path = os.path.join(
                        cx_image_path, retrieved_image + ".nii.gz." + format
                    )
                else:
                    retrieved_image_path = os.path.join(
                        cx_image_path, retrieved_image + "." + format
                    )
            else:
                retrieved_image_path = os.path.join(cx_image_path, retrieved_image)

            if retrieved_image_path.endswith(".dcm"):
                normalized_image = io.normalize_image_to_uint8(
                    io.read_dicom_to_array(retrieved_image_path),
                    percentiles=percentiles,
                )
            elif retrieved_image_path.endswith(".png"):
                normalized_image = io.normalize_image_to_uint8(
                    np.array(Image.open(retrieved_image_path)), percentiles=percentiles
                )
            plt.subplot(num_labels, 6, idx * 6 + i + 2)
            if overlay:
                plt.imshow(normalized_image)
            else:
                plt.imshow(normalized_image, cmap="gray")
            # Compute the label category for the retrieved image
            label_category = train_features_df[
                train_features_df["Name"] == retrieved_image
            ]["Label Category"].values[0]

            # Set the title for the retrieved image
            plt.title(f"Rank: {i + 1}\nLabel: {label}\nCategory: {label_category}")
            plt.axis("off")

    plt.show()


class FeatureDatasetSearch(data.Dataset):
    """
    A custom PyTorch dataset class to handle the features and metadata for image search tasks.

    This class is designed to load image feature vectors along with their associated metadata (such as image names and labels).
    It can handle different modes (train, val, test) and provides the necessary data for model training or evaluation.

    Parameters:
    -----------
    data_dict : dict
        A dictionary containing the dataset information. It must include the following keys:
        - `img_name`: List or array of image file names (strings).
        - `features`: List or array of feature vectors (each corresponding to an image).
        - `Label` (optional): List or array of ground truth labels for the images (for training/validation modes).

    mode : str, optional (default="train")
        The mode in which the dataset is being used. It can be one of the following:
        - `"train"`: Used for training, where the labels are provided.
        - `"val"`: Used for validation, where the labels are provided.
        - `"test"`: Used for testing, where labels are typically not available.

    Attributes:
    -----------
    mode : str
        The mode of the dataset (train, val, or test).
    img_name : list
        A list of image file names.
    features : numpy.ndarray
        A numpy array containing the feature vectors for each image.
    Label : numpy.ndarray or None
        A numpy array containing the ground truth labels for each image (optional in test mode).
    """

    def __init__(self, data_dict, mode="train"):
        self.mode = mode
        self.img_name = data_dict["img_name"]
        self.features = np.array(data_dict["features"], dtype="float32")
        self.Label = data_dict.get(
            "Label", None
        )  # Handle cases where Label might be missing

    def __getitem__(self, idx):
        """
        Fetches a sample from the dataset at the specified index.

        Parameters:
        -----------
        idx : int
            The index of the sample to retrieve.

        Returns:
        --------
        Tuple
            A tuple containing:
            - The feature vector (numpy array).
            - The label (int) for training/validation modes or None for test mode.
            - The image name (string).
        """
        features = self.features[idx]
        img_name = self.img_name[idx]

        if self.mode in ["train", "val"]:
            label = np.array(
                self.Label[idx], dtype=np.int64
            ).squeeze()  # Ensure proper label shape
            return features, label, img_name
        return features, img_name

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
        int
            The number of samples (images) in the dataset.
        """
        return len(self.img_name)


def create_data_loader_from_df(
    df, mode="test", batch_size=1, num_workers=2, pin_memory=True
):
    """
    Creates a data loader for the generated embeddings, including the preparation of sample data
    from the provided DataFrame.

    This function combines the data preparation and data loading into one step.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing the dataset. It should include at least the following columns:
        - `mi2_features`: The feature vectors corresponding to each image.
        - `Name`: The image file names.
        - `Label`: The ground truth labels for each image (optional for testing).

    mode : str, optional (default='test')
        The mode for the dataset. This can be one of the following:
        - `"train"`: Use for training datasets.
        - `"val"`: Use for validation datasets.
        - `"test"`: Use for testing datasets. Labels are optional in test mode.

    batch_size : int, optional (default=1)
        The batch size for the data loader. Determines how many samples are loaded per iteration.

    num_workers : int, optional (default=2)
        The number of subprocesses to use for data loading. Increasing this value can speed up data loading when using a large dataset.

    pin_memory : bool, optional (default=True)
        Whether to pin memory for faster data transfer to the GPU during training. Set to `True` to improve performance when working with large datasets on the GPU.

    Returns:
    --------
    torch.utils.data.DataLoader
        A PyTorch DataLoader object that handles the batching and loading of samples during training, validation, or testing.

    Example:
    --------
    train_loader = create_data_loader_from_df(train_df, mode="train", batch_size=32)
    """
    # Prepare the samples
    samples = {
        "features": df["mi2_features"].tolist(),
        "img_name": df["Name"].tolist(),
        "Label": df["Label"].tolist(),  # Label is optional for test mode
    }

    # Create the dataset
    dataset = FeatureDatasetSearch(samples, mode)

    # Create and return the DataLoader
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def perform_inference_and_return_features(model, test_loader):
    """
    Perform inference on the test dataset using the provided model and return the predictions along with extracted features.

    This function runs the model in evaluation mode, performs inference on the test dataset using the provided `test_loader`,
    applies a softmax function to get the class probabilities, and then returns a list of predictions, which includes the
    image name, predicted class, associated probability, and extracted feature vector for each image.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to be used for inference. It should output both extracted features and classification scores.

    test_loader : torch.utils.data.DataLoader
        A DataLoader object for the test dataset. It should provide batches of image features and their corresponding image names.

    Returns:
    --------
    predictions : list of dict
        A list of dictionaries, where each dictionary contains:
        - `"Name"`: The image name.
        - `"PredictedClass"`: The predicted class label.
        - `"Probability"`: The probability of the predicted class.
        - `"Features"`: The feature vector extracted from the image.

    Example:
    --------
    predictions = perform_inference_and_return_features(model, test_loader)
    """
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        for features, img_names in tqdm(test_loader, desc="Inference", ncols=80):
            features = features.to(device)  # Move features to the device (GPU/CPU)
            extracted_features, output = model(
                features
            )  # Get features and output from the model

            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(output, dim=1)
            predicted_classes = (
                probabilities.argmax(dim=1).cpu().numpy()
            )  # Get the class with highest probability

            # Collect predictions and extracted features
            for img_name, predicted_class, prob, feature_vector in zip(
                img_names,
                predicted_classes,
                probabilities.cpu().numpy(),
                extracted_features.cpu().numpy(),
            ):
                predictions.append(
                    {
                        "Name": img_name,
                        "PredictedClass": predicted_class,
                        "Probability": prob[
                            predicted_class
                        ],  # Get the probability for the predicted class
                        "Features": feature_vector.tolist(),  # Convert tensor to list
                    }
                )
    return predictions


def check_pkl_files(df, mi2_embd_set, train_test="training"):
    # Check for existing pickle files and return a list of subjects to generate embeddings for.
    subj_list = []
    saved_embd_list = []
    for img in df[train_test]:
        subj_list.append(img[: img.index(".")])

    if os.path.exists(mi2_embd_set) == False:
        os.makedirs(mi2_embd_set)
    else:
        for pklf in os.listdir(mi2_embd_set):
            if pklf.endswith("_mi2_embedding.pkl"):
                saved_embd_list.append(pklf[: pklf.index("_mi2")])

    embds_to_generate = [x for x in subj_list if x not in saved_embd_list]

    return embds_to_generate, subj_list

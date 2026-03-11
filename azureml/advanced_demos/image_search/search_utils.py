import numpy as np
import faiss
import pandas as pd
import torch
from torch.utils import data
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
    faiss_index,
    test_df,
    train_df,
    k_list,
    query_column_feature_name="mi2_features",
    eval_label="Label",
    eval_category="Label Category",
):
    """
    Evaluate the performance of a FAISS-based nearest neighbor search.

    This function calculates the precision of Top-K retrieval for a given test dataset
    and generates three data frames:
    1. summary_df: Overall precision for each specified k.
    2. detail_df: Label-wise precision with detailed breakdown for each k.
    3. search_results_df: Detailed search results for each query.

    Parameters:
    -----------
    faiss_index : faiss.Index
        A trained FAISS index for nearest neighbor search.
    test_df : pandas.DataFrame
        Test dataset containing:
            - Column specified by `query_column_feature_name` (default: "mi2_features"): Feature vectors stored as NumPy arrays.
            - `Label`: Ground truth labels for evaluation.
            - `Label Category`: Human-readable label categories.
    train_df : pandas.DataFrame
        Training dataset containing:
            - Column specified by `query_column_feature_name` (default: "mi2_features"): Feature vectors stored as NumPy arrays.
            - `Label`: Ground truth labels used to train the FAISS index.
    k_list : list of int
        A list of values of k (e.g., [1, 3, 5]) to evaluate Top-K retrieval precision.
    query_column_feature_name : str, optional
        Name of the column in `test_df` and `train_df` that contains the feature vectors (for example, "mi2_features").
        Default is "mi2_features".
    eval_label : str, optional
        Name of the column containing labels. Default is "Label".
    eval_category : str, optional
        Name of the column containing category names. Default is "Label Category".

    Returns:
    --------
    summary_df : pandas.DataFrame
        A data frame summarizing overall precision at each k.
        Columns:
            - `k (Top-K)`: The value of k.
            - `Overall Precision`: The average precision across all labels for the given k.
    detail_df : pandas.DataFrame
        A data frame providing label-wise precision at each k.
        Columns:
            - `Label`: Ground truth label.
            - `Category`: Human-readable label category.
            - `Precision @ k=1`: Precision at k=1.
            - `Precision @ k=3`: Precision at k=3.
            - `Precision @ k=5`: Precision at k=5, etc., depending on k_list.
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
    _, I = faiss_index.search(test_features, kmax)

    # Precompute the labels for the training features
    train_labels = train_df[eval_label].values

    # Initialize label precision
    label_precision = {
        label: {f"precision_at_{k}": [] for k in k_list}
        for label in test_df[eval_label].unique()
    }

    # Initialize a list to store search results for each query
    search_results = []

    # Iterate over each test sample
    for i in range(I.shape[0]):
        query_label = test_df.iloc[i][eval_label]
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

        # Calculate precision for each k in k_list
        for k in k_list:
            precision_at_k = np.sum(retrieved_labels[:k] == query_label) / k
            label_precision[query_label][f"precision_at_{k}"].append(precision_at_k)

    # Prepare data for summary and detail data frames
    summary_data = []
    detail_data = []

    for label in sorted(test_df[eval_label].unique()):
        if eval_category and eval_category in test_df.columns:
            label_category = test_df[test_df[eval_label] == label][eval_category].iloc[
                0
            ]
            label_data = {"Label": label, "Category": label_category}
        else:
            label_data = {"Label": label, "Category": ""}
        for k in k_list:
            avg_precision = np.mean(label_precision[label][f"precision_at_{k}"])
            label_data[f"Precision @ k={k}"] = avg_precision
        detail_data.append(label_data)

    for k in k_list:
        overall_precision = np.mean(
            [
                np.mean(label_precision[label][f"precision_at_{k}"])
                for label in label_precision
            ]
        )
        summary_data.append({"k (Top-K)": k, "Overall Precision": overall_precision})

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
    num_labels=5,
    eval_label="Label",
    eval_category=None,
    cmap=None,
    percentiles=(0.1, 0.99),
):
    """
    Display query images alongside their retrieved similar images from a search.

    This function creates a visualization grid showing query images in the first column
    and their top retrieved matches in subsequent columns. Each image is displayed with
    its label and optionally its category information.

    Parameters
    ----------
    query_df : pandas.DataFrame
        DataFrame containing query image information. Must have a "Name" column with
        image filenames. May also contain label and category columns.
    search_results_df : pandas.DataFrame
        DataFrame containing search results with columns "query_name", "retrieved_indices",
        and "retrieved_labels".
    cx_image_path : str
        Base path to the directory containing the image files.
    train_features_df : pandas.DataFrame
        DataFrame containing training set image information. Must have a "Name" column
        and may contain label/category columns for retrieved images.
    num_labels : int, optional
        Maximum number of query images (rows) to display. Default is 5.
    eval_label : str, optional
        Column name to use for label information. Default is "Label".
    eval_category : str, optional
        Column name to use for category information. If None, will attempt to use
        "Label Category" if present in both DataFrames. Default is None.
    cmap : str or None, optional
        Colormap for displaying images. If None or "gray", images are displayed in
        grayscale. Default is None.
    percentiles : tuple of float, optional
        Percentile range (low, high) for image normalization. Default is (0.1, 0.99).

    Returns
    -------
    None
        Displays the matplotlib figure directly.

    Notes
    -----
    - Supports DICOM (.dcm) and PNG (.png) image formats, with fallback for other formats.
    - The visualization grid has 6 columns: 1 for query + 5 for retrieved images.
    - Images are normalized to uint8 using the specified percentiles.
    """

    plt.figure(figsize=(20, 5 * num_labels), dpi=72)

    category_col = eval_category
    if category_col is None:
        if (
            "Label Category" in query_df.columns
            and "Label Category" in train_features_df.columns
        ):
            category_col = "Label Category"

    show_gray = (cmap is None) or (cmap == "gray")

    for idx, query_sample in query_df.iterrows():
        if idx >= num_labels:
            break

        query_results = search_results_df[
            search_results_df["query_name"] == query_sample["Name"]
        ]
        if query_results.empty:
            continue

        query_image_path = os.path.join(cx_image_path, query_sample["Name"])

        plt.subplot(num_labels, 6, idx * 6 + 1)

        if query_image_path.endswith(".dcm"):
            normalized_image = io.normalize_image_to_uint8(
                io.read_dicom_to_array(query_image_path),
                percentiles=percentiles,
            )
        elif query_image_path.endswith(".png"):
            normalized_image = io.normalize_image_to_uint8(
                np.array(Image.open(query_image_path)),
                percentiles=percentiles,
            )
        else:
            normalized_image = io.normalize_image_to_uint8(
                np.array(Image.open(query_image_path)),
                percentiles=percentiles,
            )

        # Display (gray by default)
        if show_gray:
            plt.imshow(normalized_image, cmap="gray")
        else:
            plt.imshow(normalized_image)

        # Title (ensure Category shows when available)
        label_val = (
            query_sample[eval_label]
            if eval_label in query_sample
            else query_sample.get("Label", "")
        )
        if category_col and category_col in query_sample:
            plt.title(
                f"Query\nLabel: {label_val}\nCategory: {query_sample[category_col]}"
            )
        else:
            plt.title(f"Query\nLabel: {label_val}")
        plt.axis("off")

        # Retrieved mapping
        retrieved_images = query_results["retrieved_indices"].apply(
            lambda indices: [train_features_df.iloc[i]["Name"] for i in indices]
        )

        # Retrieved loop (top 5 to fit 6 columns)
        for i, (retrieved_image, label) in enumerate(
            zip(retrieved_images.iloc[0], query_results["retrieved_labels"].iloc[0])
        ):
            if i >= 5:
                break

            retrieved_image_path = os.path.join(cx_image_path, retrieved_image)

            # Load + normalize
            if retrieved_image_path.endswith(".dcm"):
                normalized_image = io.normalize_image_to_uint8(
                    io.read_dicom_to_array(retrieved_image_path),
                    percentiles=percentiles,
                )
            elif retrieved_image_path.endswith(".png"):
                normalized_image = io.normalize_image_to_uint8(
                    np.array(Image.open(retrieved_image_path)),
                    percentiles=percentiles,
                )
            else:
                normalized_image = io.normalize_image_to_uint8(
                    np.array(Image.open(retrieved_image_path)),
                    percentiles=percentiles,
                )

            plt.subplot(num_labels, 6, idx * 6 + i + 2)

            if show_gray:
                plt.imshow(normalized_image, cmap="gray")
            else:
                plt.imshow(normalized_image)

            # Retrieved category title
            if category_col and category_col in train_features_df.columns:
                match = train_features_df[train_features_df["Name"] == retrieved_image]
                if not match.empty:
                    label_category = match[category_col].values[0]
                    plt.title(
                        f"Rank: {i + 1}\nLabel: {label}\nCategory: {label_category}"
                    )
                else:
                    plt.title(f"Rank: {i + 1}\nLabel: {label}")
            else:
                plt.title(f"Rank: {i + 1}\nLabel: {label}")

            plt.axis("off")

    plt.show()


def display_query_and_retrieved_images_radpath(
    query_df,
    search_results_df,
    path_image_path,
    rad_image_path,
    train_features_df,
    num_labels=5,
    eval_label="Label",
):
    """
    Display query pathology images alongside retrieved images and their corresponding radiology contrasts.

    This function visualizes image search results by showing a query pathology image,
    its top retrieved pathology matches, and the associated radiology images (4 contrast
    sequences) for each retrieved result in a grid layout.

    Parameters
    ----------
    query_df : pandas.DataFrame
        DataFrame containing query samples with columns 'Name' (image filename) and
        the evaluation label column.
    search_results_df : pandas.DataFrame
        DataFrame containing search results with columns 'query_name', 'retrieved_indices',
        and 'retrieved_labels'.
    path_image_path : str
        Path to the directory containing pathology images.
    rad_image_path : str
        Path to the directory containing radiology images.
    train_features_df : pandas.DataFrame
        DataFrame containing training features with columns 'Name' (image filename)
        and 'rad_id' (radiology identifier).
    num_labels : int, optional
        Maximum number of retrieved images to display per query. Default is 5.
    eval_label : str, optional
        Column name in query_df used for the evaluation label/category. Default is "Label".

    Returns
    -------
    None
        Displays a matplotlib figure with the query and retrieved images.

    Notes
    -----
    The function expects radiology images to have suffixes '_0000.nii.gz.png',
    '_0001.nii.gz.png', '_0002.nii.gz.png', and '_0003.nii.gz.png' representing
    4 different contrast sequences. Missing radiology images are indicated with
    a "Missing" label in the plot.
    """

    plt.figure(figsize=(20, 12), dpi=72)

    for idx, query_sample in query_df.iterrows():
        query_results = search_results_df[
            search_results_df["query_name"] == query_sample["Name"]
        ]
        if query_results.empty:
            continue

        retrieved_indices = query_results["retrieved_indices"].iloc[0]
        retrieved_labels = query_results["retrieved_labels"].iloc[0]

        # Query pathology image
        query_image_path = os.path.join(path_image_path, query_sample["Name"])
        query_img = Image.open(query_image_path).convert("RGB")

        plt.subplot(5, 6, 1)
        plt.imshow(query_img)
        plt.title(f"Query\nCategory: Grade {int(query_sample[eval_label])}")
        plt.axis("off")

        for rank, (retrieved_idx, retrieved_label) in enumerate(
            zip(retrieved_indices, retrieved_labels)
        ):
            if rank >= num_labels:
                break

            row = train_features_df.iloc[retrieved_idx]
            retrieved_name = row["Name"]

            # Retrieved pathology
            retrieved_path = os.path.join(path_image_path, retrieved_name)
            ret_img = Image.open(retrieved_path).convert("RGB")

            plt.subplot(5, 6, rank + 2)
            plt.imshow(ret_img)
            plt.title(f"Rank: {rank + 1}\nCategory: Grade {int(retrieved_label)}")
            plt.axis("off")

            rad_id = row["rad_id"]

            for contrast_row, suffix in enumerate(
                [
                    "_0000.nii.gz.png",
                    "_0001.nii.gz.png",
                    "_0002.nii.gz.png",
                    "_0003.nii.gz.png",
                ],
                start=1,
            ):
                contrast_path = os.path.join(rad_image_path, rad_id + suffix)

                plt.subplot(5, 6, rank + 6 * contrast_row + 2)
                if not os.path.exists(contrast_path):
                    plt.title("Missing")
                    plt.axis("off")
                    continue

                c_img = Image.open(contrast_path).convert("RGB")
                plt.imshow(c_img)
                plt.axis("off")

    plt.tight_layout()
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
    df,
    mode="test",
    category="Label",
    batch_size=1,
    num_workers=2,
    pin_memory=True,
):
    """
    Create a PyTorch DataLoader from a DataFrame containing image features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature vectors. Must have a 'Name' column and one of:
        'mi2_features', 'gigapath_features', or 'features' columns.
    mode : str, optional
        Dataset mode: 'train', 'val', or 'test'. Default is 'test'.
    category : str, optional
        Column name for labels. Default is 'Label'.
    batch_size : int, optional
        Batch size for the DataLoader. Default is 1.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 2.
    pin_memory : bool, optional
        Whether to pin memory for faster GPU transfer. Default is True.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader wrapping the feature dataset.

    Raises
    ------
    KeyError
        If df does not contain a recognized features column.
    """
    if "mi2_features" in df.columns:
        features_col = "mi2_features"
    elif "gigapath_features" in df.columns:
        features_col = "gigapath_features"
    elif "features" in df.columns:
        features_col = "features"
    else:
        raise KeyError(
            "Expected df to contain either 'mi2_features', 'gigapath_features', or 'features' column."
        )

    # Pick label column (category) if available; for test mode it may be missing
    if category in df.columns:
        labels = df[category].tolist()
    else:
        # Keep behavior tolerant for test mode; dataset can ignore labels depending on mode
        labels = [None] * len(df)

    samples = {
        "features": df[features_col].tolist(),
        "img_name": df["Name"].tolist(),
        "Label": labels,
    }

    dataset = FeatureDatasetSearch(samples, mode)

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
    """
    Check for existing pickle files and return a list of subjects to generate embeddings for.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing image file names in the specified column.
    mi2_embd_set : str
        Path to the directory where embedding pickle files are stored.
    train_test : str, optional
        Column name in df containing image file names. Default is "training".

    Returns
    -------
    tuple
        A tuple containing:
        - embds_to_generate (list): List of subject IDs that need embeddings generated.
        - subj_list (list): Complete list of all subject IDs from the input DataFrame.
    """
    subj_list = []
    saved_embd_list = []
    for img in df[train_test]:
        subj_list.append(img[: img.index(".")])

    if not os.path.exists(mi2_embd_set):
        os.makedirs(mi2_embd_set)
    else:
        for pklf in os.listdir(mi2_embd_set):
            if pklf.endswith("_mi2_embedding.pkl"):
                saved_embd_list.append(pklf[: pklf.index("_mi2")])

    embds_to_generate = [x for x in subj_list if x not in saved_embd_list]

    return embds_to_generate, subj_list


def adaptive_pooling(features, output_size):
    """
    Perform adaptive average pooling on the given features.

    Parameters
    ----------
    features : np.ndarray
        The input features to be pooled.
    output_size : int
        The spatial size of the output after pooling.

    Returns
    -------
    np.ndarray
        The pooled features with reduced spatial dimensions.
    """
    pooled_features = torch.nn.AdaptiveAvgPool2d((output_size, output_size))(
        torch.tensor(features)
    )
    pooled_features = torch.squeeze(pooled_features, dim=(2, 3))
    pooled_features = torch.squeeze(pooled_features, dim=0).data.cpu().numpy()
    return pooled_features

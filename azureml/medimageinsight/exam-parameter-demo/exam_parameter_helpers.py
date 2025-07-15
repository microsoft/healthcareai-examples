# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Exam Parameter Detection notebook helper functions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os
import pickle
import json
from openai import AzureOpenAI
from scipy.spatial import distance

from healthcareai_toolkit import settings


def plot_parameter_distribution_categorical(df, parameter_name, plot_title, height=5):
    """Use matplotlib to plot parameter distribution"""
    df[parameter_name].value_counts(dropna=False).plot(
        kind="barh", figsize=(8, height), color="#86bf91", zorder=2, width=0.9
    )
    plt.title(plot_title)

    # Create labels for the plot and include NaN values
    labels = df[parameter_name].value_counts(dropna=False)
    for i, v in enumerate(labels):
        clr = "black"
        if pd.isnull(labels.index[i]):
            clr = "red"

        plt.text(v + 1, i, str(v), color=clr, va="center")


def plot_parameter_distribution_histogram(
    df, parameter_name, plot_title, bin_count, logscale=False
):
    """Use matplotlib to plot a histogram of data"""
    # Plot the histogram directly using pandas
    df[parameter_name].hist(bins=bin_count, edgecolor="black", log=logscale)

    # Customize the plot
    plt.title(plot_title)
    plt.xlabel(parameter_name)
    plt.ylabel("Frequency")
    plt.grid(False)

    # Show the plot
    plt.show()


def sample_holdout_set(df, param_name, param_values, n_sample=5):
    """Samples a subset from the dataframe based on the parameter values provided"""
    out_pd = pd.DataFrame()
    for v in param_values:
        sampled = df[
            df[param_name].isnull() if v == None else df[param_name] == v
        ].sample(n_sample, random_state=42, replace=True)
        out_pd = pd.concat([out_pd, sampled])

    return out_pd


def create_exam_param_struct_from_dicom_tags(df_item):
    """Pack DICOM fields into a JSON object that can be sent to GPT"""
    exam_params = {}
    exam_params["Body Part Examined"] = df_item["BodyPartExamined"]
    exam_params["Protocol Name"] = df_item["ProtocolName"]
    exam_params["Series Description"] = df_item["SeriesDescription"]
    exam_params["Image Type"] = df_item["ImageType"]
    exam_params["Sequence Variant"] = df_item["SequenceVariant"]

    return json.dumps(exam_params)


def create_openai_client():
    endpoint = settings.AZURE_OPENAI_ENDPOINT
    api_key = settings.AZURE_OPENAI_API_KEY

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    return client


def create_oai_assistant(client):
    """Creates assistant to keep track of prior responses"""
    # Assistant API example: https://github.com/openai/openai-python/blob/main/examples/assistant.py
    # Available in limited regions
    deployment = (settings.AZURE_OPENAI_MODEL_NAME,)
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a categorizer. For each question answered, extract entities related to people's names and "
        " jobs and categorize them. You always return result in JSON. You reuse categories from past responses when possible",
        model=deployment,
        tools=[{"type": "code_interpreter"}],
    )
    return assistant.id


def plot_clusters(df, column):
    # Get unique values for the column
    unique_values = df[column].unique()

    # Generate a colormap with the required number of colors
    cmap = plt.get_cmap("tab20")  # You can choose any colormap you prefer
    colors = cmap(np.linspace(0, 1, len(unique_values)))

    if len(unique_values) > len(colors):
        raise ValueError(
            "Not enough colors for the number of unique values in the column"
        )

    # Create a figure and axis with a wider aspect ratio
    fig, ax = plt.subplots(figsize=(16, 7))

    for protocol, color in zip(unique_values, colors):
        subset = df[df[column] == protocol]
        ax.scatter(
            subset["embedding_p1"],
            subset["embedding_p2"],
            color=color,
            label=protocol,
            alpha=0.6,
            edgecolors="w",
            s=100,
        )

    # Create a layout for the plot
    ax.set_title("MedImageInsight dimensionality reduction vs " + column + " (MRI).")
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")

    # Move the legend outside the plot
    ax.legend(title=column, ncol=1, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show the figure
    fig.show()


def prepare_feature_maps(df, dataset_root=""):
    """Sets up feature maps for series based on MedImageInsight's individual image embeddings"""
    feat_mean_matrix = []
    feat_std_matrix = []

    for index, row in df.iterrows():
        feat_path = row["features"]
        feat_file = os.path.join(dataset_root, feat_path)
        feat_dict = pd.read_pickle(feat_file)

        # Sort features by slice number
        feat_dict = dict(sorted(feat_dict.items()))

        # creat list of dict items values
        feat_list = list(feat_dict.values())

        # Select the center slice + 10 slices before and after
        center_slice = int(len(feat_dict) / 2)
        slice_range = list(range(center_slice - 10, center_slice + 10))

        # print("number of slices: ", len(feat_dict))
        # Extract features for the selected slices and create a 2D matrix
        feat_subject_matrix = []

        # Check if slice_range is within the number of slices
        slice_range = [x for x in slice_range if x < len(feat_list)]

        for slice_num in slice_range:
            feat_subject_matrix.append(feat_list[slice_num])
        feat_subject_matrix = np.array(feat_subject_matrix)

        # Calculate the mean and standard deviation of each feature
        feat_mean = np.mean(feat_subject_matrix, axis=0)
        feat_std = np.std(feat_subject_matrix, axis=0)

        feat_center_slice = feat_list[center_slice]

        # save feat_mean as pickle file
        feat_mean_path = feat_file.replace(".pkl", ".mean.pkl")
        with open(feat_mean_path, "wb") as handle:
            pickle.dump(feat_mean, handle)

        # save feat_center_slice as pickle file
        feat_center_slice_path = feat_file.replace(".pkl", ".center_slice.pkl")
        with open(feat_center_slice_path, "wb") as handle:
            pickle.dump(feat_center_slice, handle)

        feat_mean_matrix.append(feat_mean)
        feat_std_matrix.append(feat_std)

    feat_mean_matrix = np.array(feat_mean_matrix)

    # Not used yet
    feat_std_matrix = np.array(feat_std_matrix)

    return feat_mean_matrix


def read_image(path):
    """Reads an image from a file and returns it as a np array"""
    return mpimg.imread(path)


def plot_image(df, parameter, dataset_root=""):
    # load the image and display it
    para_vals = df[parameter].dropna().unique()
    dict_para = {}

    # Calculate the number of rows needed based on the number of unique values and columns
    n_cols = 5
    n_rows = int(np.ceil(len(para_vals) / n_cols))

    # Create a subplots of images for each value of the parameter
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axs = axs.flatten()

    fig.suptitle("MRI images for each " + parameter)
    plt.subplots_adjust(wspace=0.1, hspace=0.25)  # Adjust margins

    for i, para in enumerate(para_vals):
        # Select a row from the DataFrame corresponding each value of the given column
        row = df[df[parameter] == para].iloc[0]
        # Load the image

        # list files in row['png_path'] and select the middle one
        png_path = row["png_path"]
        if dataset_root != "":
            png_path = os.path.join(dataset_root, png_path)

        files = os.listdir(png_path)
        files.sort()
        png_file = png_path + os.sep + files[len(files) // 2]

        img = read_image(png_file)
        # Display the image
        axs[i].imshow(img, cmap="gray")
        axs[i].set_title(para)
        axs[i].axis("off")
        dict_para[para] = img

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axs[j].axis("off")

    plt.show()


def display_closest_images(
    df, column, image_series, text_flag=False, images_path="", features_path=""
):
    """Display the top 5 closest images to a given image given a dataframe and a given column"""

    # The following code randomly picks an image with a given value and uses it to find the closest images
    # random_index = np.random.randint(0, len(df[df[column] == value]))

    # print('Random index: ', random_index)

    # # Select a row from the dataframe corresponding each value of of the given column
    # row = df[df[column] == value].iloc[random_index]

    # The following code uses supplied dataframe row to find closest images
    row = image_series

    pickle_path = row["features"].replace(".pkl", ".center_slice.pkl")
    if features_path != "":
        pickle_path = os.path.join(features_path, pickle_path)

    image_feat_mean = pd.read_pickle(pickle_path)

    # Compute text if text_flag is True

    if text_flag:
        # compare the image_feat_mean with the values of text_parameter_feat_dict
        image_text_similarity_list = []
        for key, value in text_parameter_feat_dict.items():
            # Cross product between the image_feat_mean and the values of text_parameter_feat_dict
            cross_product = np.dot(image_feat_mean, value)
            image_text_similarity_list.append(cross_product)

        # Create a dataframe with the similarity values
        df_distances = pd.DataFrame(
            {
                "parameter": list(text_parameter_feat_dict.keys()),
                "distance": image_text_similarity_list,
            }
        )

        # Sort the dataframe by distance
        df_distances.sort_values(by="distance", inplace=True)

        # Print the top 5 df_distances values
        print("Top 5 closest exam parameters based on text similarity:")
        print(df_distances.head(5))

    embedding_p1 = row["embedding_p1"]
    embedding_p2 = row["embedding_p2"]

    png_path = row["png_path"]
    if images_path != "":
        png_path = os.path.join(images_path, png_path)

    files = os.listdir(png_path)
    files.sort()
    png_file = png_path + os.sep + files[len(files) // 2]
    img = read_image(png_file)

    # String containing the value of the parameter of interest or "Not present" if the value is not present
    value_string = (
        row[column] if isinstance(row[column], str) and row[column] else "Not present"
    )
    plt.imshow(img, cmap="gray")
    plt.title(f"{column}: " + value_string)

    # Deep copy the dataframe
    df_copy = df.copy()

    distances = []
    for index, row in df.iterrows():
        dist = distance.euclidean(
            [embedding_p1, embedding_p2], [row["embedding_p1"], row["embedding_p2"]]
        )
        distances.append(dist)

    df_copy["distance"] = distances
    df_copy.sort_values(by="distance", inplace=True)

    # Display the top 5 closest images but not the original image
    fig, axs = plt.subplots(1, 5, figsize=(20, 10))

    for i in range(5):
        row = df_copy.iloc[i + 1]
        png_path = row["png_path"]
        if images_path != "":
            png_path = os.path.join(images_path, png_path)
        files = os.listdir(png_path)
        files.sort()
        png_file = png_path + os.sep + files[len(files) // 2]
        img = read_image(png_file)
        axs[i].imshow(img, cmap="gray")
        value_string = (
            row[column]
            if isinstance(row[column], str) and row[column]
            else "Not present"
        )
        axs[i].set_title(
            f"{column}: " + value_string + "\n" + "Distance: " + str(row["distance"])
        )
    fig.show()

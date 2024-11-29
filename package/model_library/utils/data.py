# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from torch.utils import data


class FeatureDataset(data.Dataset):
    def __init__(self, data_dict, csv, mode="train"):
        self.data_dict = data_dict
        self.csv = csv
        self.mode = mode
        self.img_name = data_dict["img_name"]
        self.features = data_dict["features"]

    def __getitem__(self, item):
        img_name = self.img_name[item]
        features = self.features[item]
        features = features.astype("float32")

        row = self.csv[self.csv["Name"] == img_name]
        if self.mode == "train" or self.mode == "val":
            label = row["Label"].values

            label = np.array(label)
            label = np.reshape(label, (1,))
            label = label.squeeze()

            return features, label, img_name

        elif self.mode == "test":
            return features, img_name

    def __len__(self):
        return len(self.img_name)

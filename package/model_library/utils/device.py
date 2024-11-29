# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import warnings


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

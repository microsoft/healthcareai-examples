# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from tqdm import tqdm
import warnings

from .device import get_device


def perform_inference(model, test_loader):

    predictions = []
    device = get_device()
    model.eval()
    with torch.no_grad():
        for features, img_names in tqdm(test_loader, desc="Inference", ncols=80):
            features = features.to(device)
            _, output = model(features)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            predicted_classes = probabilities.argmax(dim=1).cpu().numpy()
            # Collect predictions
            for img_name, predicted_class, prob in zip(
                img_names, predicted_classes, probabilities.cpu().numpy()
            ):
                predictions.append(
                    {
                        "Name": img_name,
                        "PredictedClass": predicted_class,
                        "Probability": prob[predicted_class],
                    }
                )
    return predictions


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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from tqdm import tqdm

from .device import get_device


def perform_inference(model, test_loader):

    predictions = []
    device = get_device()
    model.eval()
    with torch.no_grad():
        for features, label, img_names in tqdm(test_loader, desc="Inference", ncols=80):
            features = features.to(device)
            _, output = model(features)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(output, dim=1)
            predicted_classes = probabilities.argmax(dim=1).cpu().numpy()
            # Collect predictions
            for img_name, predicted_class, prob, sample_label in zip(
                img_names, predicted_classes, probabilities.cpu().numpy(), label
            ):
                predictions.append(
                    {
                        "Name": img_name,
                        "PredictedClass": predicted_class,
                        "Probability": prob[predicted_class],
                        "Label": sample_label.item(),
                    }
                )
    return predictions


def load_trained_model(model, model_path):
    # Load Model State
    device = get_device()
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    return model

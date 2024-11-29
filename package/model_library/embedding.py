# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn


class AdapterModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_class):
        super().__init__()

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_class = num_class

        ## Adaptor Module
        self.vision_embd = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )

        self.retrieval_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
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
        self.prediction_head = nn.Sequential(nn.Linear(self.hidden_dim, self.num_class))

    def forward(self, vision_feat):
        feat = self.vision_embd(vision_feat.squeeze(1))
        feat = self.retrieval_conv(torch.unsqueeze(feat, 2))
        class_output = self.prediction_head(feat.squeeze(2))

        return feat, class_output

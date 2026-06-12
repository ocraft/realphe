from typing import cast

import torch
from torch import nn


class GaussianNoise(nn.Module):

    def __init__(self, std):
        super().__init__()

        std = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("std", std)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * cast(torch.Tensor, self.std)
            return x + noise

        return x

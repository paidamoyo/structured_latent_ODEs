import torch.nn as nn
import torch


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)

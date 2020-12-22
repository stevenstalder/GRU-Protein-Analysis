import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

class CNNClassificationHead(nn.Module):

    def __init__(self,
                 in_dim: int = 2048,
                 hid_dim: int = 512,
                 out_dim: int = 3,
                 dropout: float = 0.):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous()
        return x
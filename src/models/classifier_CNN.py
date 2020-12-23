import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from utils.argparser import *

class Classifier_CNN(nn.Module):

    def __init__(self):
        super().__init__()
        parser = get_parser()
        self.hparams = parser.parse_args()

        self.cnn = nn.Sequential(
            nn.BatchNorm1d(self.hparams.gru_hidden_out_size),  # Added this
            weight_norm(nn.Conv1d(self.hparams.gru_hidden_out_size, self.hparams.cnn_hidden_size, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(self.hparams.cnn_dropout, inplace=True),
            weight_norm(nn.Conv1d(self.hparams.cnn_hidden_size, self.hparams.num_classes, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2).contiguous()
        return x
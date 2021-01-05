import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from utils.argparser import *

"""
Defines the autoregressive classification head. CNN architecture with two inputs:
1. Amino acid encoding, which is processed by a convolutional layer
2. Previous sequence prediction, processed by masked convolution to make sure that only information available from the past
is learned
The results of these convolutions are then concatenated and processed by another final convolution layer
Masked convolutions based on PixelCNN, specifically the implementation at https://github.com/singh-hrituraj/PixelCNN-Pytorch
"""

class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, length = self.weight.size()
        self.mask.fill_(1)
        self.mask[:,:,length//2:] = 0


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedConv1d, self).forward(x)


class Classifier_autoregressive(nn.Module):

    def __init__(self):
        super().__init__()
        parser = get_parser()
        self.hparams = parser.parse_args()

        first_conv_layer = nn.Conv1d(self.hparams.enc_hidden_out_size, self.hparams.cnn_hidden_size, 5, dilation=2, padding=4) if self.hparams.cnn_dilated \
                                else nn.Conv1d(self.hparams.enc_hidden_out_size, self.hparams.cnn_hidden_size, 5, padding=2)

        first_conv_layer_autoreg = MaskedConv1d(1, self.hparams.cnn_hidden_size, 5, dilation=2, padding=4) if self.hparams.cnn_dilated \
                                else MaskedConv1d(1, self.hparams.cnn_hidden_size, 5, padding=2)

        self.cnn = nn.Sequential(
            nn.BatchNorm1d(self.hparams.enc_hidden_out_size),
            weight_norm(first_conv_layer, dim=None),
            nn.ReLU(),
            nn.Dropout(self.hparams.cnn_dropout, inplace=True))

        self.autoreg = nn.Sequential(    
            #  
            #     
            #
            weight_norm(first_conv_layer_autoreg, dim=None),
            nn.ReLU(),
            nn.Dropout(self.hparams.cnn_dropout, inplace=True))

        self.combined = nn.Sequential(
            weight_norm(nn.Conv1d(self.hparams.cnn_hidden_size*2, self.hparams.num_classes, 3, padding=1), dim=None))

    def forward(self, x, y):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        batchsize, seqlength = y.size()
        y = y.view(batchsize, 1, seqlength).float()
        y = self.autoreg(y)
        xy = torch.cat((x, y), dim=1)
        xy = self.combined(xy)
        xy = xy.transpose(1, 2).contiguous()
        return xy
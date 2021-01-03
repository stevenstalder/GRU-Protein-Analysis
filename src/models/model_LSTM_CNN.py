import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl

import numpy as np
from test_tube import Experiment, HyperOptArgumentParser

from utils.argparser import *
from utils.tokenizer import *
from utils.accuracy import *
from models.encoder_LSTM import *
from models.classifier_CNN import *

"""
Initialization of the PyTorch Lightning model combining the LSTM based encoder and the CNN based classification head.
To account for different length protein sequences when computing the accuracy, we changed the way they are accumulated 
for the validation and test set by defining validation_epoch_end(), test_epoch_end (). Cross Entropy Loss and Accuracy are 
computed on a padded sequence with the padding value of -1 and then ignored in the computation.
"""

class Protein_LSTM_Sequencer_CNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        parser = get_parser()
        self.hparams = parser.parse_args()

        self.Encoder = Encoder_LSTM()

        self.Classifier = Classifier_CNN()

    def forward(self, x):
        encoding = self.Encoder(x)
        output = self.Classifier(encoding)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        l = self(x)
        y = pad_sequence(y, batch_first=True, padding_value=-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        acc_fct = TrainAccuracy(ignore_index=-1)
        acc = acc_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        self.log('loss', loss, on_epoch=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        l = self(x)
        y = pad_sequence(y, batch_first=True, padding_value=-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        acc_fct = TestAccuracy(ignore_index=-1)
        acc = acc_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        self.log('test_loss', loss, on_epoch=True)
        
        return acc

    def test_epoch_end(self, test_step_outputs):
        total = 0.0
        total_correct = 0.0
        for acc in test_step_outputs:
            correct, valid = acc
            total_correct += correct
            total += valid

        accuracy = total_correct / total
        self.log('test_acc', accuracy, on_epoch=True, prog_bar=True)    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        l = self(x)
        y = pad_sequence(y, batch_first=True, padding_value=-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        acc_fct = TestAccuracy(ignore_index=-1)
        acc = acc_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        self.log('val_loss', loss, on_epoch=True)

        return acc

    def validation_epoch_end(self, validation_step_outputs):
        total = 0.0
        total_correct = 0.0
        for acc in validation_step_outputs:
            correct, valid = acc
            total_correct += correct
            total += valid

        accuracy = total_correct / total
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

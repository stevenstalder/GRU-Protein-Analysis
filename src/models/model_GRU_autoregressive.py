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
from models.encoder_GRU import *
from models.classifier_autoregressive import *


"""
Initialization of the PyTorch Lightning model combining the GRU based encoder and the autoregressive classification head.
During training, one of the inputs is the target sequence, with masked convolutions making sure that no information is leaked.
Validation and testing run the model multiple times in an autoregressive fashion (initial input is seeded randomly). 
The autoregressive_steps hyperparameter controls how many amino acid structure predictions are taken from each run during
validation and testing before being fed back to predict the next group.
Computation of accuracy is done the same way as for the other models.
"""

class Protein_GRU_Sequencer_Autoregressive(pl.LightningModule):
    def __init__(self):
        super().__init__()

        parser = get_parser()
        self.hparams = parser.parse_args()

        self.Encoder = Encoder_GRU()

        self.Classifier = Classifier_autoregressive()

    def forward(self, x, y):
        encoding = self.Encoder(x)
        output = self.Classifier(encoding, y)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        l = self(x, y)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        acc_fct = TrainAccuracy(ignore_index=-1)
        acc = acc_fct(l.view(-1, self.hparams.num_classes), y.view(-1))

        self.log('loss', loss, on_epoch=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return loss
    #
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = pad_sequence(y, batch_first=True, padding_value=-1)

        #l = self(x,y).clone().fill_(0)
        l = (torch.rand(self(x,y).size())*3).cuda()
        stepsize = self.hparams.autoregressive_steps
        for i in range(0, l.size(1), stepsize):
            l_temp = self(x, l.argmax(-1))
            l[:,i:i+stepsize] = l_temp[:,i:i+stepsize]

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
        y = pad_sequence(y, batch_first=True, padding_value=-1)
        

        #l = self(x,y).clone().fill_(0)
        l = (torch.rand(self(x,y).size())*3).cuda()
        stepsize = self.hparams.autoregressive_steps
        for i in range(0, l.size(1), stepsize):
            l_temp = self(x, l.argmax(-1))
            l[:,i:i+stepsize] = l_temp[:,i:i+stepsize]

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
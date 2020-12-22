import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import numpy as np
from test_tube import Experiment, HyperOptArgumentParser

from utils.sequenceclassifier import *
from utils.argparser import *
from utils.tokenizer import *

class Net_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        parser = get_parser()
        self.hparams = parser.parse_args()

        self.gru = nn.GRU(
            input_size=self.hparams.gru_input_size,
            hidden_size=self.hparams.gru_hidden_size,
            batch_first=True,
            num_layers=self.hparams.gru_layers,
            bidirectional=self.hparams.gru_bidirectional,
            dropout=self.hparams.gru_dropout,
        )
        self.hidden_lin_size = (
            self.hparams.gru_hidden_size
            * (self.hparams.gru_bidirectional + 1)
        )
        self.linear = nn.Linear(self.hidden_lin_size,
                                self.hparams.gru_hidden_out_size)

    def forward(self, x):
        outputs, _ = self.gru(x)
        # outputs: [B, A, num_direction * E]
        y = self.linear(outputs)
        return y



class Encoder_GRU(nn.Module):
    def __init__(self):
        super().__init__()

        parser = get_parser()
        self.hparams = parser.parse_args()
        self.tokenizer = TAPETokenizer(vocab="iupac")
        self.token_emb = nn.Embedding(self.hparams.vocab_size, self.hparams.gru_input_size)

        self.gru = Net_GRU()

    def forward(self, x):
        protein_encoded = [torch.tensor(self.tokenizer.encode(item).tolist()) for item in np.asarray(x)]
        protein_encoded_tensor = torch.tensor(pad_sequence(protein_encoded, batch_first=True))

        embeddings = self.token_emb(protein_encoded_tensor)
        output = self.gru(embeddings)
        return output


class Protein_GRU_Sequencer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        parser = get_parser()
        self.hparams = parser.parse_args()

        self.Encoder = Encoder_GRU()

        self.Classifier = Classifier_CNN(hidden_size=524, num_labels=1)

    def forward(self, x):
        encoding = self.Encoder(x)
        output = self.Classifier(encoding)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        # 1. Forward pass
        l = self(x)
        
        # 2. Loss
        loss = F.cross_entropy(l, y.type(torch.LongTensor))

        acc = accuracy(l, y.type(torch.LongTensor))

        self.log('loss', loss, on_epoch=True)
        self.log('acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        l = self(x)
        
        loss = F.cross_entropy(l, y.type(torch.LongTensor))
        acc = accuracy(l, y.type(torch.LongTensor))

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        l = self(x)
        l = torch.tensor(pad_sequence(l, batch_first=True))
        y = torch.tensor(pad_sequence(y, batch_first=True))

        loss_fct = nn.CrossEntropyLoss(
                    ignore_index=0)

        classification_loss = loss_fct(
            l.view(-1, 1), y.view(-1))
        acc_fct = Accuracy(ignore_index=0)
        acc = acc_fct(l.view(-1, 1), y.view(-1))

        self.log('val_loss', classification_loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.hparams.learning_rate)
        return optimizer


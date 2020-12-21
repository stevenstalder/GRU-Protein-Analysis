import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from test_tube import Experiment, HyperOptArgumentParser

from utils.sequenceclassifier import *


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hparams = parser.parse_args()

        self.gru = nn.GRU(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            batch_first=True,
            num_layers=self.hparams.gru_layers,
            bidirectional=self.hparams.bidirectional,
            dropout=self.hparams.gru_dropout,
        )
        self.hidden_out_size = (
            self.hparams.hidden_size
            * (self.hparams.bidirectional + 1)
        )
        self.linear = nn.Linear(self.hidden_out_size, 1)

    def forward(self, x):
        outputs, _ = self.gru(x)
        # outputs: [B, T, num_direction * H]
        y = self.linear(outputs)
        return y



class GRUEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.hparams = parser.parse_args()

        self.token_emb = nn.Embedding(
            self.hparams.vocab_size, self.hparams.input_size)

        self.gru = GRUNet()

    def forward(self, x):
        embeddings = self.token_emb(x)
        output, _ = self.gru(embeddings)
        return output


class Protein_GRU_Sequencer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.Encoder = GRUEncoder()

        self.Classifier = SequenceToSequenceClassificationHead()

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

        loss = F.cross_entropy(l, y.type(torch.LongTensor))
        acc = accuracy(l, y.type(torch.LongTensor))

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.hparams.learning_rate)
        return optimizer


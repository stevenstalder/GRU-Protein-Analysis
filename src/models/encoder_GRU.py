import torch
from torch import nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import numpy as np
from test_tube import Experiment, HyperOptArgumentParser

from utils.argparser import *
from utils.tokenizer import *

class Net_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        parser = get_parser()
        self.hparams = parser.parse_args()

        self.gru = nn.GRU(
            input_size=self.hparams.enc_input_size,
            hidden_size=self.hparams.enc_hidden_size,
            batch_first=True,
            num_layers=self.hparams.enc_layers,
            bidirectional=self.hparams.enc_bidirectional,
            dropout=self.hparams.enc_dropout,
        )
        self.hidden_lin_size = (
            self.hparams.enc_hidden_size
            * (self.hparams.enc_bidirectional + 1)
        )
        self.linear = nn.Linear(self.hidden_lin_size,
                                self.hparams.enc_hidden_out_size)

    def forward(self, x):
        outputs, _ = self.gru(x)
        # outputs: [Batch, AminoAcid, num_direction * EmbeddingDim]
        y = self.linear(outputs)
        return y



class Encoder_GRU(nn.Module):
    def __init__(self):
        super().__init__()

        parser = get_parser()
        self.hparams = parser.parse_args()
        self.tokenizer = TAPETokenizer(vocab=self.hparams.tokenizer)
        self.token_emb = nn.Embedding(self.hparams.vocab_size, self.hparams.enc_input_size)

        self.gru = Net_GRU()

    def forward(self, x):
        protein_encoded = [torch.tensor(self.tokenizer.encode(item).tolist()) for item in np.asarray(x)]
        protein_encoded_tensor = pad_sequence(protein_encoded, batch_first=True)
        if torch.cuda.is_available():
            protein_encoded_tensor = protein_encoded_tensor.cuda()

        embeddings = self.token_emb(protein_encoded_tensor)
        output = self.gru(embeddings)
        return output




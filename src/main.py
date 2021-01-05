import datetime
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from __init__ import *

pl.seed_everything(hparams.seed)

"""
Call all scripts from __init__, define and initialize model, set and activate Tensorboard logging and run training, validation and testing loop.
early stopping set fix to 10 rounds of no improvement of at least 0.001 validation accuracy.
"""

parser = get_parser()
hparams = parser.parse_args()


# Define Early Stopping condition
early_stop_callback = EarlyStopping(
   monitor='val_acc',
   min_delta=0.001,
   patience=10,
   verbose=False,
   mode='max'
)

# Define Model
if hparams.classifier_type == "autoregressive":
    model = Protein_GRU_Sequencer_Autoregressive()
elif hparams.encoder_type == "gru":
    model = Protein_GRU_Sequencer_CNN()
elif hparams.encoder_type == "lstm":
    model = Protein_LSTM_Sequencer_CNN()
else:
    raise Exception('Unknown encoder type: ' + hparams.encoder_type)


# Set Logging
if hparams.logger == True:
   log_dir = "tb_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   logger = pl.loggers.TensorBoardLogger(log_dir, name='Protein_SS3_Model')
   # How to launch Tensorboard: https://www.tensorflow.org/tensorboard/get_started
else:
   logger = False

# Train, validate and test Model
trainer = pl.Trainer(logger=logger, callbacks=[early_stop_callback], gpus=torch.cuda.device_count())
trainer.fit(model, train_loader, val_loader)

trainer.test(test_dataloaders=test_loader)

import datetime
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from __init__ import *

parser = get_parser()
hparams = parser.parse_args()

pl.seed_everything(hparams.seed)

"""
Pytorch Lightning training, validation and testing here. Maybe with visualizations.
"""

### Define Early Stopping condition ###
early_stop_callback = EarlyStopping(
   monitor='val_acc',
   min_delta=0.001,
   patience=10,
   verbose=False,
   mode='max'
)

### Define Model ###
model = Protein_GRU_Sequencer()


# Set Logging

if hparams.logger == True:
   log_dir = "tb_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   logger = pl.loggers.TensorBoardLogger(log_dir, name='Protein_SS3_Model')
else:
   logger = False

# tensorboard --logdir # To start tensorboard on local port 6006

### Train Model ###
trainer = pl.Trainer(logger=logger, callbacks=[early_stop_callback], gpus=torch.cuda.device_count())
trainer.fit(model, train_loader, val_loader)

trainer.test(test_dataloaders=test_loader)

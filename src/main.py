import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from __init__ import *

"""
Pytorch Lightning training, validation and testing here. Maybe with visualizations.
"""

### Define Early Stopping condition ###
early_stop_callback = EarlyStopping(
   monitor='val_acc',
   min_delta=0.001,
   patience=2,
   verbose=False,
   mode='max'
)

### Define Model ###
model = Protein_GRU_Sequencer()

### Train Model ###
use_gpu = 1 if torch.cuda.device_count() > 0 else 0
trainer = pl.Trainer(logger=False, callbacks=[early_stop_callback], gpus=use_gpu)
trainer.fit(model, train_loader, val_loader)


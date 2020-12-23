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
   min_delta=0.0001,
   patience=20,
   verbose=False,
   mode='max'
)

### Define Model ###
model = Protein_GRU_Sequencer()

### Train Model ###
trainer = pl.Trainer(logger=False, callbacks=[early_stop_callback], gpus=torch.cuda.device_count())
trainer.fit(model, train_loader, val_loader)

trainer.test(test_dataloaders=test_loader)

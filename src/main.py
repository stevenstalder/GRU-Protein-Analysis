import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
from __init__ import *

"""
Pytorch Lightning training, validation and testing here. Maybe with visualizations.
"""

model = Protein_GRU_Sequencer()

trainer = pl.Trainer(logger=False)
trainer.fit(model, train_loader, val_loader)


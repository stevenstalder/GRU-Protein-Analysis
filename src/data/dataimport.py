import torch
from pathlib import Path
from data.download import *
from data.dataloader import *

"""
Import dataloaders and downloads data if respective folders are empty.
"""

base_path = Path(__file__).resolve().parents[2]
data_path = base_path / "tape_data"

get_tape_data()

### Import Dataloaders ###

test = JsonDataset(data_path / "test" / "test.json")
test_loader = DataLoader(test, batch_size=32, collate_fn=collate_fn)

train = JsonDataset(data_path / "training" / "train.json")
train_loader = DataLoader(train, batch_size=32, collate_fn=collate_fn)

val = JsonDataset(data_path / "validation" / "val.json")
val_loader = DataLoader(val, batch_size=32, collate_fn=collate_fn)


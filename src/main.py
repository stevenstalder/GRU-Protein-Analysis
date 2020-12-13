import torch
from pathlib import Path
from data.download import *
from data.dataloader import *

base_path = Path(__file__).resolve().parents[1]
data_path = base_path / "tape_data"

get_tape_data()

### Testing stuff ###
dataset = JsonDataset(data_path / "test" / "test.json")
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for batch in dataloader:
    X, y = batch
    print(X)
    print(y)
    break


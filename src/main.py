import torch
from data.download import *
from data.dataloader import *

get_tape_data()

### Testing stuff ###
dataset = JsonDataset('../tape_data/test/test.json')
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

for batch in dataloader:
    X, y = batch
    print(X)
    print(y)
    break


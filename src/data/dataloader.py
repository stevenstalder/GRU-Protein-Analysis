import torch
import json
from torch.utils.data import DataLoader, IterableDataset


class JsonDataset(IterableDataset):
    def __init__(self, json_file):
        self.file = json_file

    def __iter__(self):
    	with open(self.file) as f:
    		data = json.load(f)
    		for sample in data:
    			yield sample['primary'], sample['ss3']

    def __len__(self):
        with open(self.file) as f:
            data = json.load(f)
            return len(data)

#change this to output what we want (most importantly character encoding)
def collate_fn(batch):
	primary = tuple([item[0] for item in batch])
	labels = tuple([torch.LongTensor(item[1]) for item in batch])
	return [primary, labels]
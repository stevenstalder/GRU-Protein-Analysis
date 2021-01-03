import torch
from torch import nn

"""
Accuracy classes for training (TrainAccuracy) and validation + testing (TestAccuracy)
"""

class TrainAccuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return train_accuracy(inputs, target, self.ignore_index)

class TestAccuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return test_accuracy(inputs, target, self.ignore_index)

### Outputs accuracy of batch directly ###
def train_accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

### Outputs (num_correct, total_valid) for each batch which is then accumulated in validation_epoch_end or test_epoch_end ###
def test_accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return (correct.sum().float(),valid_mask.sum().float())

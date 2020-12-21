from configargparse import ArgumentParser
import argparse

'''
This file contains the declaration of our argument parser

GRU: (for later)
input_size: int = 128, hidden_size: int = 1024, gru_layers: int = 3,
                 hidden_out_size: int = 768, gru_dropout: float = 0.2, 
                 bidirectional: bool = False,  vocab_size: int = 30,
                 num_classes: int = 2, learning_rate: float = 1e-4
'''

def get_parser():
    parser = ArgumentParser(description='Protein SS3 Task',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--learning_rate", default=1e-4, type=float)

    return parser

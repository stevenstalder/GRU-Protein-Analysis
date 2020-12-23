from argparse import ArgumentParser

'''
This file contains the declaration of our argument parser
'''

def get_parser():
    parser = ArgumentParser(description='Protein SS3 Task')
    #Trainer Parameter
    
    # General Model Parameter
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    # GRU Encoder
    parser.add_argument("--gru_input_size", default=128, type=int)
    parser.add_argument("--gru_hidden_size", default=1024, type=int)
    parser.add_argument("--gru_layers", default=3, type=int)
    parser.add_argument("--gru_hidden_out_size", default=726, type=int)
    parser.add_argument("--gru_dropout", default=0.2, type=float)
    parser.add_argument("--gru_bidirectional", default=False, type=bool)
    parser.add_argument("--vocab_size", default=30, type=int)


    return parser

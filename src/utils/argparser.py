from argparse import ArgumentParser

'''
This file contains the declaration of our argument parser
'''

# Needed to parse booleans from command line properly
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = ArgumentParser(description='Protein SS3 Task')
    # Trainer Parameters
    
    # General Model Parameters
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    # GRU Encoder Parameters
    parser.add_argument("--gru_input_size", default=128, type=int)
    parser.add_argument("--gru_hidden_size", default=1024, type=int)
    parser.add_argument("--gru_layers", default=3, type=int)
    parser.add_argument("--gru_hidden_out_size", default=2048, type=int)
    parser.add_argument("--gru_dropout", default=0.2, type=float)
    parser.add_argument("--gru_bidirectional", default=False, type=str2bool)
    parser.add_argument("--vocab_size", default=30, type=int)
    parser.add_argument("--tokenizer", choices=["iupac", "unirep"], default="iupac", type=str)
    # CNN Classifier Parameters
    parser.add_argument("--cnn_hidden_size", default=512, type=int)
    parser.add_argument("--cnn_dropout", default=0.0, type=float)
    parser.add_argument("--num_classes", default=3, type=int)

    return parser

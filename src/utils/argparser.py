from argparse import ArgumentParser

"""
This file contains the declaration of our argument parser
"""

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
    parser.add_argument("--logger", default=False, type=str2bool)
    # General Model Parameters
    parser.add_argument("--encoder_type", choices=["gru", "lstm"], default="gru", type=str)
    parser.add_argument("--classifier_type", choices=["cnn", "autoregressive"], default="cnn", type=str)
    parser.add_argument("--autoregressive_steps", default=50, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    # GRU/LSTM Encoder Parameters
    parser.add_argument("--enc_input_size", default=128, type=int)
    parser.add_argument("--enc_hidden_size", default=1024, type=int)
    parser.add_argument("--enc_layers", default=3, type=int)
    parser.add_argument("--enc_hidden_out_size", default=2048, type=int)
    parser.add_argument("--enc_dropout", default=0.2, type=float)
    parser.add_argument("--enc_bidirectional", default=False, type=str2bool)
    parser.add_argument("--vocab_size", default=30, type=int)
    parser.add_argument("--tokenizer", choices=["iupac", "unirep"], default="iupac", type=str)
    # CNN Classifier Parameters
    parser.add_argument("--cnn_dilated", default=False, type=str2bool)
    parser.add_argument("--cnn_hidden_size", default=512, type=int)
    parser.add_argument("--cnn_dropout", default=0.0, type=float)
    parser.add_argument("--num_classes", default=3, type=int)

    return parser

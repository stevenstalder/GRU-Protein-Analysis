import typing
import copy
import json
import logging
import os
from io import open
import math
from torch.nn.utils.weight_norm import weight_norm

import torch
from torch import nn
import torch.nn.functional as F

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin" # No need when we dont want to save the weights

logger = logging.getLogger(__name__)

from utils.sequenceclassifier import *

"""
All classes needed to construct the LSTM Encoder Model with sequence classification head. We can simplify this 
by a lot. We dont need the "ProteinConfig" object since we just have one model and dont need logging etc.
"""



class ProteinConfig(object): # Get rid of this!
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods
        for loading/downloading/saving configurations.
        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names`
                (string) as keys and `url` (string) of associated pretrained model
                configurations as values.
        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune
                the model.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is
                a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns
                attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all
                hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~ProteinConfig.from_pretrained`
            class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the " \
                                              "model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~ProteinConfig`
             (or a derived class) from a pre-trained model configuration.
        Parameters:
            pretrained_model_name_or_path: either:
                - a string with the `shortcut name` of a pre-trained model configuration to
                  load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~ProteinConfig.save_pretrained` method,
                  e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`,
                  e.g.: ``./my_model_directory/configuration.json``.
            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            kwargs: (`optional`) dict:
                key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will
                  be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration
                  attributes is controlled by the `return_unused_kwargs` keyword parameter.
            return_unused_kwargs: (`optional`) bool:
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)`
                  where `unused_kwargs` is a dictionary consisting of the key/value pairs
                  whose keys are not configuration attributes: ie the part of kwargs which
                  has not been used to update `config` and is otherwise ignored.
        Examples::
            # We can't instantiate directly the base class `ProteinConfig` so let's
              show the examples on a derived class: ProteinBertConfig
            # Download configuration from S3 and cache.
            config = ProteinBertConfig.from_pretrained('bert-base-uncased')
            # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = ProteinBertConfig.from_pretrained('./test/saved_model/')
            config = ProteinBertConfig.from_pretrained(
                './test/saved_model/my_configuration.json')
            config = ProteinBertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True,
                foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}
        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(
                pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path( # Ignore it. We dont need it
                config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model "
                             "configuration file.".format(config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            return None
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP: typing.Dict[str, str] = {} # Get rid of this
LSTM_PRETRAINED_MODEL_ARCHIVE_MAP: typing.Dict[str, str] = {}

class ProteinLSTMConfig(ProteinConfig):
    pretrained_config_archive_map = LSTM_PRETRAINED_CONFIG_ARCHIVE_MAP # No need since we dont pretrain

    def __init__(self,
                 vocab_size: int = 30,
                 input_size: int = 128,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 3,
                 hidden_dropout_prob: float = 0.1,
                 initializer_range: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range


class ProteinLSTMAbstractModel(ProteinModel):

    config_class = ProteinLSTMConfig
    pretrained_model_archive_map = LSTM_PRETRAINED_MODEL_ARCHIVE_MAP # No need since we dont pretrain
    base_model_prefix = "lstm"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ProteinLSTMLayer(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, inputs):
        inputs = self.dropout(inputs)
        self.lstm.flatten_parameters()
        return self.lstm(inputs)


class ProteinLSTMPooler(nn.Module): # No need IF we dont work with the pooled output
    def __init__(self, config):
        super().__init__()
        self.scalar_reweighting = nn.Linear(2 * config.num_hidden_layers, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.scalar_reweighting(hidden_states).squeeze(2)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinLSTMEncoder(nn.Module):

    def __init__(self, config: ProteinLSTMConfig):
        super().__init__()
        forward_lstm = [ProteinLSTMLayer(
            config.input_size, config.hidden_size)]
        reverse_lstm = [ProteinLSTMLayer(
            config.input_size, config.hidden_size)]
        for _ in range(config.num_hidden_layers - 1):
            forward_lstm.append(ProteinLSTMLayer(
                config.hidden_size, config.hidden_size, config.hidden_dropout_prob))
            reverse_lstm.append(ProteinLSTMLayer(
                config.hidden_size, config.hidden_size, config.hidden_dropout_prob))
        self.forward_lstm = nn.ModuleList(forward_lstm)
        self.reverse_lstm = nn.ModuleList(reverse_lstm)
        self.output_hidden_states = config.output_hidden_states

    def forward(self, inputs, input_mask=None):
        all_forward_pooled = ()
        all_reverse_pooled = ()
        all_hidden_states = (inputs,)
        forward_output = inputs
        for layer in self.forward_lstm:
            forward_output, forward_pooled = layer(forward_output)
            all_forward_pooled = all_forward_pooled + (forward_pooled[0],)
            all_hidden_states = all_hidden_states + (forward_output,)

        reversed_sequence = self.reverse_sequence(inputs, input_mask)
        reverse_output = reversed_sequence
        for layer in self.reverse_lstm:
            reverse_output, reverse_pooled = layer(reverse_output)
            all_reverse_pooled = all_reverse_pooled + (reverse_pooled[0],)
            all_hidden_states = all_hidden_states + (reverse_output,)
        reverse_output = self.reverse_sequence(reverse_output, input_mask)

        output = torch.cat((forward_output, reverse_output), dim=2)
        pooled = all_forward_pooled + all_reverse_pooled
        pooled = torch.stack(pooled, 3).squeeze(0)
        outputs = (output, pooled)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        return outputs  # sequence_embedding, pooled_embedding, (hidden_states)

    def reverse_sequence(self, sequence, input_mask):
        if input_mask is None:
            idx = torch.arange(sequence.size(1) - 1, -1, -1)
            reversed_sequence = sequence.index_select(
                1, idx, device=sequence.device)
        else:
            sequence_lengths = input_mask.sum(1)
            reversed_sequence = []
            for seq, seqlen in zip(sequence, sequence_lengths):
                idx = torch.arange(seqlen - 1, -1, -1, device=seq.device)
                seq = seq.index_select(0, idx)
                seq = F.pad(seq, [0, 0, 0, sequence.size(1) - seqlen])
                reversed_sequence.append(seq)
            reversed_sequence = torch.stack(reversed_sequence, 0)
        return reversed_sequence


class ProteinLSTMModel(ProteinLSTMAbstractModel):

    def __init__(self, config: ProteinLSTMConfig):
        super().__init__(config)
        self.embed_matrix = nn.Embedding(config.vocab_size, config.input_size)
        self.encoder = ProteinLSTMEncoder(config)
        self.pooler = ProteinLSTMPooler(config)
        self.output_hidden_states = config.output_hidden_states
        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # fp16 compatibility
        embedding_output = self.embed_matrix(input_ids)
        outputs = self.encoder(embedding_output, input_mask=input_mask)
        sequence_output = outputs[0]
        pooled_outputs = self.pooler(outputs[1])

        outputs = (sequence_output, pooled_outputs) + outputs[2:]
        return outputs  # sequence_output, pooled_output, (hidden_states)

class ProteinLSTMForSequenceToSequenceClassification(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        self.classify = SequenceToSequenceClassificationHead(
            config.hidden_size * 2, config.num_labels, ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        amino_acid_class_scores = self.classify(sequence_output.contiguous())

        # add hidden states and if they are here
        outputs = (amino_acid_class_scores,) + outputs[2:]

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            classification_loss = loss_fct(
                amino_acid_class_scores.view(-1, self.config.num_labels),
                targets.view(-1))
            outputs = (classification_loss,) + outputs

        # (loss), prediction_scores, seq_relationship_score, (hidden_states)
        return outputs

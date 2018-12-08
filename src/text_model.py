import tensorflow as tf
import numpy as np

import os
from pathlib import Path

from self_attention import SelfAttention
from tensorflow import keras

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense


class Config():
    def __init__(
            self,
            *,
            lstm_layers,
            lstm_size,
            embedding_size,
            feature_params,
            attention_num_heads,
            attention_head_size,
            use_pretrained_embeddings=False,
            embedding_regularization_coef=0.0,
            dense_regularization_coef=0.0,
            lstm_dropout=0.0,
            collect_metrics=True,
    ):
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.attention_head_size = attention_head_size
        self.attention_num_heads = attention_num_heads
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.lstm_dropout = lstm_dropout
        self.embedding_regularization_coef = embedding_regularization_coef
        self.dense_regularization_coef = dense_regularization_coef
        self.collect_metrics = collect_metrics
        self.vocab_size = feature_params['vocab_size']
        self.num_classes = 4


class Model(keras.Model):
    def __init__(self, config):
        super(Model, self).__init__()
        regularizer = keras.regularizers.l2(config.embedding_regularization_coef)
        if config.use_pretrained_embeddings:
            weights = _load_character_embeddings()
            self.embedding_layer = Embedding(config.vocab_size,
                                             config.embedding_size,
                                             weights=[weights],
                                             trainable=False,
                                             mask_zero=True)
        else:
            self.embedding_layer = Embedding(config.vocab_size,
                                             config.embedding_size,
                                             embeddings_regularizer=regularizer,
                                             mask_zero=True)
        self.recurrent_layer = LSTM(config.lstm_size,
                                    recurrent_dropout=config.lstm_dropout,
                                    return_sequences=True)
        self.attention_layer = SelfAttention(config.attention_num_heads,
                                             config.attention_head_size)
        dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
        self.dense_layer = Dense(config.num_classes * 8,
                                 kernel_regularizer=dense_regularizer,
                                 activation='relu')
        self.output_layer = Dense(config.num_classes,
                                  activation=tf.nn.softmax)

    def call(self, inputs):
        tokens = inputs#['tokens']
        embedded = self.embedding_layer(tokens)
        recurrent_out = self.recurrent_layer(embedded)
        attended = self.attention_layer(recurrent_out)
        # print_op = tf.print(tokens)
        # with tf.control_dependencies([print_op]):
        probs = self.output_layer(self.dense_layer(attended))
        return probs


class SimpleModel(keras.Model):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        regularizer = keras.regularizers.l2(config.embedding_regularization_coef)
        self.embedding_layer = Embedding(config.vocab_size,
                                         config.embedding_size,
                                         embeddings_regularizer=regularizer,
                                         mask_zero=True)
        self.recurrent_layer = LSTM(config.lstm_size, return_state=True)
        self.concat_layer = Concatenate()
        dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
        self.dense_layer = Dense(config.num_classes * 8,
                                 kernel_regularizer=dense_regularizer,
                                 activation='relu')
        self.output_layer = Dense(config.num_classes,
                                  activation=tf.nn.softmax)

    def call(self, inputs):
        # tokens = inputs['tokens']
        tokens = inputs
        embedded = self.embedding_layer(tokens)
        _, state_1, state_2 = self.recurrent_layer(embedded)
        recurrent_out = self.concat_layer([state_1, state_2])
        # print_op = tf.print(tokens)
        # with tf.control_dependencies([print_op]):
        probs = self.output_layer(self.dense_layer(recurrent_out))
        return probs


def _get_data_paths():
    resource_path = Path(os.environ['AI_RESOURCE_PATH'])
    char_embedding_path = resource_path / 'glove.840B.300d-char.txt'
    word_embedding_path = resource_path / 'glove.840B.300d.txt'
    return {'char_embedding_path': char_embedding_path,
            'word_embedding_path': word_embedding_path}


def _load_character_embeddings() -> np.ndarray:
    data_paths = _get_data_paths()
    result = np.ndarray((255, 300))
    with open(data_paths['char_embedding_path'], mode='r') as f:
        for l in f:
            tokens = l.split(' ')
            char = tokens[0]
            idx = ord(char)
            values = [float(v) for v in tokens[1:]]
            result[idx] = values
    return result


def _load_word_embeddings() -> np.ndarray:
    data_paths = _get_data_paths()
    intern_dict = {}
    embeddings = np.ndarray((2196017, 300))
    with open(data_paths['word_embedding_path'], mode='r') as f:
        for l in f:
            tokens = l.split(' ')
            word = tokens[0]
            idx = len(intern_dict)
            intern_dict[word] = idx
            values = [float(v) for v in tokens[1:]]
            embeddings[idx] = values
    return (intern_dict, embeddings)

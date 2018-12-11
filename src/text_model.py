import tensorflow as tf
import numpy as np

import os
from pathlib import Path
from typing import Optional

from self_attention import SelfAttention
from tensorflow import keras

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant


class Config():
    def __init__(
            self,
            *,
            lstm_size,
            embedding_size,
            feature_params,
            attention_num_heads=None,
            attention_head_size=None,
            use_attention=True,
            use_pretrained_embeddings=False,
            use_word_level_embeddings=False,
            glove_vocab_size=200000,
            embedding_regularization_coef=0.0,
            dense_regularization_coef=0.0,
            lstm_dropout=0.0,
            dense_dropout=0.0,
            collect_metrics=True,
    ):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.attention_head_size = attention_head_size
        self.attention_num_heads = attention_num_heads
        self.use_attention = use_attention
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.use_word_level_embeddings = use_word_level_embeddings
        self.glove_vocab_size = glove_vocab_size
        self.lstm_dropout = lstm_dropout
        self.embedding_regularization_coef = embedding_regularization_coef
        self.dense_regularization_coef = dense_regularization_coef
        self.dense_dropout = dense_dropout
        self.collect_metrics = collect_metrics
        self.vocab_size = feature_params['vocab_size']
        self.num_classes = 4

        if use_attention and (attention_head_size is None or attention_num_heads is None):
            raise TypeError


class Model(keras.Model):
    def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
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
        dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
        if config.use_word_level_embeddings:
            if checkpoint_path is None:
                weights = _load_word_embeddings(config.glove_vocab_size)[1]
                embed_init = Constant(weights)
            else:
                embed_init = None
            self.word_embedding_layer = Embedding(config.glove_vocab_size + 1,
                                                  300,
                                                  embeddings_initializer=embed_init,
                                                  trainable=False,
                                                  mask_zero=True)
            self.word_embedding_dropout = Dropout(config.dense_dropout)
            self.word_embbedding_attention = SelfAttention(2, 64)
            self.word_dense_h_1 = Dense(config.lstm_size,
                                        activation='relu',
                                        kernel_regularizer=dense_regularizer)
            self.word_dense_h_2 = Dense(config.lstm_size,
                                        activation='tanh',
                                        kernel_regularizer=dense_regularizer)
            self.word_dense_c_1 = Dense(config.lstm_size,
                                        activation='relu',
                                        kernel_regularizer=dense_regularizer)
            self.word_dense_c_2 = Dense(config.lstm_size,
                                        activation='tanh',
                                        kernel_regularizer=dense_regularizer)
        self.recurrent_layer = LSTM(config.lstm_size,
                                    recurrent_dropout=config.lstm_dropout,
                                    return_state=not config.use_attention,
                                    return_sequences=config.use_attention)
        if config.use_attention:
            self.attention_layer = SelfAttention(config.attention_num_heads,
                                                 config.attention_head_size)
        self.attention_dropout = Dropout(config.dense_dropout)
        self.dense_layer = Dense(config.num_classes * 8,
                                 kernel_regularizer=dense_regularizer,
                                 activation='relu')
        self.output_layer = Dense(config.num_classes,
                                  activation=tf.nn.softmax)
        self.config = config
        if checkpoint_path is not None:
            self.load_weights(checkpoint_path)

    def call(self, inputs, training=False, return_attention_weights=False):
        if self.config.use_word_level_embeddings:
            tokens, word_tokens, uncased_word_tokens = inputs
            tokens = tf.cast(tokens, tf.int64)
            word_tokens = self._fixup_tokens(word_tokens,
                                             self.config.glove_vocab_size)
            uncased_word_tokens = self._fixup_tokens(uncased_word_tokens,
                                                     self.config.glove_vocab_size)
            word_embedded = tf.concat([self.word_embedding_layer(word_tokens),
                                       self.word_embedding_layer(uncased_word_tokens)],
                                      axis=-1)
            word_embedded = self.word_embedding_dropout(word_embedded,
                                                        training=training)
            # word_embedded = tf.reduce_mean(word_embedded, axis=-2)
            word_embedded = self.word_embbedding_attention(word_embedded)
            initial_h = self.word_embedding_dropout(self.word_dense_h_1(word_embedded),
                                                    training=training)
            initial_h = self.word_dense_h_2(initial_h)
            initial_c = self.word_embedding_dropout(self.word_dense_c_1(word_embedded),
                                                    training=training)
            initial_c = self.word_dense_c_2(initial_c)
            initial_state = [initial_h, initial_c]
        else:
            tokens = inputs
            initial_state = None

        embedded = self.embedding_layer(tokens)
        recurrent_out = self.recurrent_layer(embedded,
                                             initial_state=initial_state,
                                             training=training)

        if self.config.use_attention:
            attended = self.attention_layer(recurrent_out,
                                            training=training,
                                            return_attention_weights=return_attention_weights)
            if return_attention_weights:
                attended, attention_weights = attended
        else:
            _, state_h, state_c = recurrent_out
            attended = tf.concat([state_h, state_c], axis=-1)
        attended = self.attention_dropout(attended, training=training)
    
        probs = self.output_layer(self.dense_layer(attended))
        if return_attention_weights:
            return (probs, attention_weights)
        else:
            return probs

    def _fixup_tokens(self, tokens, vocab_size):
        # that a cast is necessary is a sign of some bug in Keras I think
        tokens = tf.cast(tokens, tf.int64)
        # note: the last element of the vocab is the unknown token vector
        tokens = tf.clip_by_value(tokens, 0, vocab_size)
        return tokens

    def extract_inputs_from_dict(self, input_dict):
        if self.config.use_word_level_embeddings:
            return [input_dict['tokens'],
                    input_dict['word_tokens'],
                    input_dict['uncased_word_tokens']]
        else:
            return input_dict['tokens']


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


FULL_GLOVE_VOCAB_SIZE = 2196017


def _load_word_embeddings(vocab_size=FULL_GLOVE_VOCAB_SIZE):
    data_paths = _get_data_paths()
    intern_dict = {}
    embeddings = np.zeros((vocab_size + 1, 300), dtype=np.float32)
    with open(data_paths['word_embedding_path'], mode='r') as f:
        for l in f:
            tokens = l.split(' ')
            word = tokens[0]
            idx = len(intern_dict) + 1  # 0 is the padding token
            if idx >= vocab_size:
                break
            intern_dict[word] = idx
            values = [float(v) for v in tokens[1:]]
            embeddings[idx] = values
    # a random vector for unknown tokens
    embeddings[vocab_size] = np.random.normal(0,
                                              scale=1/np.sqrt(300),
                                              size=[300])
    return (intern_dict, embeddings)

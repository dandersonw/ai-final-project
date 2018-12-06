import tensorflow as tf

from self_attention import SelfAttention
from tensorflow import keras


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
        self.embedding_layer = keras.layers.Embedding(config.vocab_size,
                                                      config.embedding_size,
                                                      embeddings_regularizer=regularizer,
                                                      mask_zero=True)
        lstm_cell = keras.layers.LSTMCell(config.lstm_size)
        self.recurrent_layer = keras.layers.RNN(lstm_cell,
                                                return_sequences=True)
        self.attention_layer = SelfAttention(config.attention_num_heads,
                                             config.attention_head_size)
        dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
        self.dense_layer = keras.layers.Dense(config.num_classes * 8,
                                              kernel_regularizer=dense_regularizer,
                                              activation='relu')
        self.output_layer = keras.layers.Dense(config.num_classes,
                                               activation=tf.nn.softmax)

    def call(self, inputs):
        tokens = inputs['tokens']
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
        self.embedding_layer = keras.layers.Embedding(config.vocab_size,
                                                      config.embedding_size,
                                                      embeddings_regularizer=regularizer,
                                                      mask_zero=True)
        lstm_cell = keras.layers.LSTMCell(config.lstm_size)
        self.recurrent_layer = keras.layers.RNN(lstm_cell, return_state=True)
        self.concat_layer = keras.layers.Concatenate()
        dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
        self.dense_layer = keras.layers.Dense(config.num_classes * 8,
                                              kernel_regularizer=dense_regularizer,
                                              activation='relu')
        self.output_layer = keras.layers.Dense(config.num_classes,
                                               activation=tf.nn.softmax)

    def call(self, inputs):
        tokens = inputs['tokens']
        embedded = self.embedding_layer(tokens)
        _, state_1, state_2 = self.recurrent_layer(embedded)
        recurrent_out = self.concat_layer([state_1, state_2])
        # print_op = tf.print(tokens)
        # with tf.control_dependencies([print_op]):
        probs = self.output_layer(self.dense_layer(recurrent_out))
        return probs

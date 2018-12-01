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


def create_model(config):
    tokens = keras.Input(shape=[None], dtype=tf.int64, name='tokens')
    # lengths = keras.Input(shape=[], dtype=tf.int64, name='length')

    embeddings_regularizer = keras.regularizers.l2(config.embedding_regularization_coef)
    embedding_layer = keras.layers.Embedding(config.vocab_size,
                                             config.embedding_size,
                                             embeddings_regularizer=embeddings_regularizer,
                                             mask_zero=True)
    embedded = embedding_layer(tokens)

    # TODO, multiple LSTM layers, multi headed attention
    # cell = keras.layers.LSTMCell(config.lstm_size,
    #                              recurrent_dropout=config.lstm_dropout)
    # recurrent_layer = keras.layers.RNN(cell, return_sequences=True)
    recurrent_layer = keras.layers.LSTM(config.lstm_size, return_sequences=True)
    recurrent_out = recurrent_layer(embedded)

    attention_layer = SelfAttention(config.attention_num_heads,
                                    config.attention_head_size)
    attended = attention_layer(recurrent_out)

    output_layer = keras.layers.Dense(config.num_classes)
    logits = output_layer(attended)

    model = keras.Model(inputs=tokens, outputs=logits)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

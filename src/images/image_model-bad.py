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
    embedded = _embed(tokens, config)
    recurrent_layer = keras.layers.LSTM(config.lstm_size,
                                        return_sequences=True)
    recurrent_out = recurrent_layer(embedded)

    attention_layer = SelfAttention(config.attention_num_heads,
                                    config.attention_head_size)
    attended = attention_layer(recurrent_out)

    logits = _output(attended, config)

    model = keras.Model(inputs=tokens, outputs=logits)

    model.compile(optimizer=tf.keras.optimizers.SGD(clipnorm=5.0),
                  loss=tf.losses.softmax_cross_entropy,
                  metrics=['accuracy'])

    return model


def create_simple_model(config):
    tokens = keras.Input(shape=[None], dtype=tf.int64, name='tokens')
    embedded = _embed(tokens, config)

    recurrent_layer = keras.layers.LSTM(config.lstm_size)
    recurrent_out = recurrent_layer(embedded)

    logits = _output(recurrent_out, config)

    model = keras.Model(inputs=tokens, outputs=logits)

    model.compile(optimizer=tf.keras.optimizers.SGD(clipnorm=5.0),
                  loss=tf.losses.softmax_cross_entropy,
                  metrics=['accuracy'])

    return model


def _embed(tokens, config):
    embeddings_regularizer = keras.regularizers.l2(config.embedding_regularization_coef)
    embedding_layer = keras.layers.Embedding(config.vocab_size,
                                             config.embedding_size,
                                             embeddings_regularizer=embeddings_regularizer,
                                             mask_zero=True)
    return embedding_layer(tokens)


def _output(inputs, config):
    dense_regularizer = keras.regularizers.l2(config.dense_regularization_coef)
    dense_layer = keras.layers.Dense(config.num_classes * 8,
                                     kernel_regularizer=dense_regularizer,
                                     activation='relu')
    output_layer = keras.layers.Dense(config.num_classes)
    return output_layer(dense_layer(inputs))

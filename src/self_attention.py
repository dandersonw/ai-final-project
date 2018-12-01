import tensorflow as tf

from tensorflow import keras


# https://arxiv.org/pdf/1703.03130.pdf

class SelfAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size

    def build(self, input_shape):
        self.W_1 = self.add_weight('W_1',
                                   dtype=tf.float32,
                                   shape=[self.head_size,
                                          input_shape[-1].value])
        self.W_2 = self.add_weight('W_2',
                                   dtype=tf.float32,
                                   shape=[self.num_heads,
                                          self.head_size])

    def compute_output_shape(self, input_shape):
        return [input_shape[:-2], self.num_heads * input_shape[-1]]

    def call(self, inputs):
        # inputs = (batch, time, in_dim)
        # weights = (head_size, batch, time)
        weights = tf.tensordot(self.W_1, inputs, [[1], [2]])
        weights = tf.tanh(weights)
        # weights = (num_heads, batch, in_dim)
        weights = tf.tensordot(self.W_2, weights, [[1], [0]])
        # weights = (batch, num_heads, in_dim)
        weights = tf.transpose(weights, perm=[1, 0, 2])
        weights = tf.nn.softmax(weights)

        # weighted = (batch, num_heads, in_dim)
        weighted = tf.matmul(weights, inputs)

        duplication = tf.tensordot(weights, weights, [[2], [2]])
        duplication = tf.transpose(duplication, perm=[0, 2, 1, 3])
        duplication -= tf.eye(tf.shape(duplication)[-1])
        norm = tf.norm(duplication,
                       axis=[-2, -1],
                       ord='fro')
        coef = 4e-3
        penalty = coef * tf.square(norm)
        self.add_loss(tf.reduce_sum(penalty))

        return tf.layers.flatten(weighted)

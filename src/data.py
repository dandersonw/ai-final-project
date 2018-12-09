import tensorflow as tf
import numpy as np

import re


IMAGE_DIMS = [50, 50, 3]
NUM_CLASSES = 4
FEATURE_KEYS = {'tokens', 'length', 'image', 'word_tokens'}


def datum_to_tf_example(datum: dict) -> tf.train.SequenceExample:
    example = tf.train.SequenceExample()
    example.context.feature['length'].int64_list.value.append(datum['length'])
    example.context.feature['label'].int64_list.value.append(datum['label'])
    # the image is expected as a bytes object, JPEG encoded
    example.context.feature['image'].bytes_list.value.append(datum['image'])
    tokens = example.feature_lists.feature_list['tokens'].feature
    for t in datum['tokens']:
        tokens.add().int64_list.value.append(t)
    word_tokens = example.feature_lists.feature_list['word_tokens'].feature
    for t in datum['word_tokens']:
        word_tokens.add().int64_list.value.append(t)
    return example


def parse_tf_example(example, features):
    context_features = {'label': tf.FixedLenFeature([], dtype=tf.int64),
                        'length': tf.FixedLenFeature([], dtype=tf.int64),
                        'image': tf.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'tokens': tf.FixedLenSequenceFeature([],
                                                              dtype=tf.int64),
                         'word_tokens': tf.FixedLenSequenceFeature([],
                                                                   dtype=tf.int64)}

    context_parsed, sequence_parsed \
        = tf.parse_single_sequence_example(context_features=context_features,
                                           sequence_features=sequence_features,
                                           serialized=example)

    decoded_image = tf.image.decode_jpeg(context_parsed['image'])
    resized_image = tf.cast(tf.round(tf.image.resize_images(decoded_image,
                                                            IMAGE_DIMS[:-1])),
                            tf.uint8)
    adjusted_label = context_parsed['label'] - 1
    one_hot = tf.one_hot(adjusted_label, NUM_CLASSES, dtype=tf.float32)

    all_features = {'tokens': sequence_parsed['tokens'],
                    'word_tokens': sequence_parsed['word_tokens'],
                    'length': context_parsed['length'],
                    'image': resized_image}

    # Not returning features we don't need saves computation time
    # In pure TF code it wouldn't matter,
    # but Keras must force evaluation at some point
    returned_features = {k: v for k, v in all_features.items() if k in features}
    return (returned_features,
            one_hot)


def make_dataset(path, batch_size=128, features=FEATURE_KEYS) -> tf.data.Dataset:
    with tf.name_scope(path):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(lambda d: parse_tf_example(d, features))
        dataset = dataset.shuffle(buffer_size=1000)
        # TODO: data augmentation?
        padding_shapes = {'length': [],
                          'tokens': [None],
                          'word_tokens': [None],
                          'image': IMAGE_DIMS}
        padding_shapes = {k: v for k, v in padding_shapes.items() if k in features}
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=(padding_shapes,
                                                      [NUM_CLASSES]))
        dataset = dataset.prefetch(10)
        return dataset


def _unintern_tokens(tokens) -> str:
    return "".join(chr(t) for t in tokens if t)


def extract_card_names_from_dataset(dataset, sess):
    result = []
    iterator = dataset.make_one_shot_iterator().get_next()
    while True:
        try:
            batch = sess.run(iterator)
            result.extend([_unintern_tokens(tokens) for tokens in batch[0]['tokens']])
        except tf.errors.OutOfRangeError:
            break
    return result


def preload_dataset(path, sess, features=FEATURE_KEYS):
    dataset = make_dataset(path, batch_size=100000, features=features)
    X = None
    y = None
    iterator = dataset.make_one_shot_iterator().get_next()
    while True:
        try:
            batch = sess.run(iterator)
            if X is None:
                X = {k: [v] for k, v in batch[0].items()}
                y = [batch[1]]
            else:
                for k in X:
                    X[k].append(batch[0][k])
                    y.append(batch[1])
        except tf.errors.OutOfRangeError:
            break
    X = {k: np.concatenate(v) for k, v in X.items()}
    y = np.concatenate(y)
    return (X, y)

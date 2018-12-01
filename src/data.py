import tensorflow as tf


def datum_to_tf_example(datum: dict) -> tf.train.SequenceExample:
    example = tf.train.SequenceExample()
    example.context.feature['length'].int64_list.value.append(datum['length'])
    example.context.feature['label'].int64_list.value.append(datum['label'])
    # the image is expected as a bytes object, JPEG encoded
    example.context.feature['image'].bytes_list.value.append(datum['image'])
    tokens = example.feature_lists.feature_list['tokens'].feature
    for t in datum['tokens']:
        tokens.add().int64_list.value.append(t)
    return example


def parse_tf_example(example):
    context_features = {'label': tf.FixedLenFeature([], dtype=tf.int64),
                        'length': tf.FixedLenFeature([], dtype=tf.int64),
                        'image': tf.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'tokens': tf.FixedLenSequenceFeature([],
                                                              dtype=tf.int64)}

    context_parsed, sequence_parsed \
        = tf.parse_single_sequence_example(context_features=context_features,
                                           sequence_features=sequence_features,
                                           serialized=example)

    decoded_image = tf.image.decode_jpeg(context_features['image'])

    # return {'label': context_parsed['label'],
    #         'length': context_parsed['length'],
    #         'image': decoded_image,
    #         'tokens': sequence_parsed['tokens']}
    return ({'length': context_parsed['length'],
             'image': decoded_image,
             'tokens': sequence_parsed['tokens']},
            context_parsed['label'])


def make_dataset(path, batch_size=128) -> tf.data.Dataset:
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.shuffle(buffer_size=10000)
    # TODO: data augmentation?
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes={'length': [],
                                                  'label': [],
                                                  'tokens': [None],
                                                  'image': []})
    dataset = dataset.prefetch(10)
    # return dataset.make_initializable_iterator()
    return dataset # with tf 1.12 this should be possible?

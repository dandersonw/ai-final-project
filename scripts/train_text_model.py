import tensorflow as tf

import argparse
import importlib
import sys

MY_SRC = './src/'
sys.path.append(MY_SRC)

import text_model
import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--config_dir', required=True)
    parser.add_argument('--config_name', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()

    sys.path.append(args.config_dir)
    config_module = importlib.import_module(args.config_name)
    config = config_module.config
    features = config_module.features

    with tf.Session() as sess:
        training_data = data.preload_dataset(args.train, sess, features=features)
        validation_data = data.preload_dataset(args.valid, sess, features=features)
        testing_data = data.preload_dataset(args.test, sess, features=features)

    model = train_model(config, training_data, validation_data, args.checkpoint_path)
    model.evaluate(x=testing_data[0]['tokens'], y=testing_data[1])


def train_model(config, training_data, validation_data, checkpoint_path):
    model = text_model.Model(config)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=4)
    compilation_kwargs = {'optimizer': tf.keras.optimizers.Adam(clipnorm=5.0),
                          'loss': 'categorical_crossentropy',
                          'metrics': ['accuracy']}
    model.compile(**compilation_kwargs)

    train_kwargs = {'x': model.extract_inputs_from_dict(training_data[0]),
                    'y': training_data[1],
                    'validation_data': (validation_data[0]['tokens'], validation_data[1]),
                    'callbacks': [checkpoint, early_stopping]}
    
    if config.use_pretrained_embeddings:
        model.fit(**train_kwargs, epochs=10)
        model.embedding_layer.trainable = True
        model.compile(**compilation_kwargs)
        model.fit(**train_kwargs, epochs=20)
    else:
        model.fit(**train_kwargs, epochs=30)

    model.load_weights(checkpoint_path)
    return model


if __name__ == '__main__':
    main()

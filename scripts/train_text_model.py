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

    _train_model(args.train,
                 args.valid,
                 args.test,
                 args.config_dir,
                 args.config_name,
                 args.checkpoint_path)


def _train_model(train_data_path,
                 valid_data_path,
                 test_data_path,
                 config_dir,
                 config_name,
                 checkpoint_path):
    sys.path.append(config_dir)
    config_module = importlib.import_module(config_name)
    config = config_module.config
    features = config_module.features

    with tf.Session() as sess:
        training_data = data.preload_dataset(train_data_path, sess, features=features)
        validation_data = data.preload_dataset(valid_data_path, sess, features=features)
        testing_data = data.preload_dataset(test_data_path, sess, features=features)

    model = train_model(config, training_data, validation_data, checkpoint_path)
    scores = model.evaluate(x=model.extract_inputs_from_dict(testing_data[0]), y=testing_data[1])
    for (metric, score) in zip(model.metrics_names, scores):
        print('{}: {:.4f}'.format(metric, score))


def train_model(config, training_data, validation_data, checkpoint_path):
    model = text_model.Model(config)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    compilation_kwargs = {'optimizer': tf.keras.optimizers.Adam(clipnorm=5.0),
                          'loss': 'categorical_crossentropy',
                          'metrics': ['accuracy']}
    model.compile(**compilation_kwargs)

    train_kwargs = {'x': model.extract_inputs_from_dict(training_data[0]),
                    'y': training_data[1],
                    'validation_data': (model.extract_inputs_from_dict(validation_data[0]),
                                        validation_data[1]),
                    'callbacks': [checkpoint, early_stopping]}
    
    if config.use_pretrained_embeddings:
        model.fit(**train_kwargs, epochs=10)
        model.embedding_layer.trainable = True
        model.compile(**compilation_kwargs)
        model.fit(**train_kwargs, epochs=40)
    else:
        model.fit(**train_kwargs, epochs=50)

    model.load_weights(checkpoint_path)
    return model


if __name__ == '__main__':
    main()

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
        testing_data = data.preload_dataset(args.test, sess, features=features)

    model = text_model.Model(config, checkpoint_path=args.checkpoint_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    scores = model.evaluate(x=model.extract_inputs_from_dict(testing_data[0]),
                            y=testing_data[1])
    for (metric, score) in zip(model.metrics_names, scores):
        print('{}: {:.4f}'.format(metric, score))


if __name__ == '__main__':
    main()

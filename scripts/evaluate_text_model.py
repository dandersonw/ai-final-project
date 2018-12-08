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

    model = text_model.Model(config)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.evaluate(x=testing_data[0]['tokens'], y=testing_data[1])

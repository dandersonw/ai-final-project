import tensorflow as tf
import pandas as pd
import numpy as np

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
    parser.add_argument('--report_path')
    args = parser.parse_args()

    evaluate(args.test, args.config_dir, args.config_name, args.checkpoint_path, args.report_path)


def evaluate(test_data_path, config_dir, config_name, checkpoint_path, report_path):
    sys.path.append(config_dir)
    config_module = importlib.import_module(config_name)
    config = config_module.config
    features = config_module.features

    with tf.Session() as sess:
        testing_data = data.preload_dataset(test_data_path, sess, features=features)

    model = text_model.Model(config, checkpoint_path=checkpoint_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    scores = model.evaluate(x=model.extract_inputs_from_dict(testing_data[0]),
                            y=testing_data[1])
    for (metric, score) in zip(model.metrics_names, scores):
        print('{}: {:.4f}'.format(metric, score))

    if report_path is not None:
        predictions = model.predict(model.extract_inputs_from_dict(testing_data[0]))
        labels = np.argmax(testing_data[1], axis=-1)
        names = [data._unintern_tokens(t) for t in testing_data[0]['tokens']]

        report = pd.DataFrame({'predicted': np.argmax(predictions, axis=-1),
                               'label': labels,
                               'name': names,
                               **{'class_{}_p'.format(i): predictions[:, i] for i in range(4)}})
        report.to_csv(report_path, index=None)


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np

import argparse
import data
import json
import pickle

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_path')
    parser.add_argument('feature_param_path')
    parser.add_argument('mode')
    args = parser.parse_args()

    if args.mode == 'train':
        feature_params = feature_params_from_datafile(args.input_file)
        pickle.dump(feature_params, open(args.feature_param_path, mode='wb'))
    else:
        feature_params = pickle.load(open(args.feature_param_path, mode='rb'))

    with tqdm(open(args.input_file)) as input_file:
        writer = tf.python_io.TFRecordWriter(args.output_path)
        raw_data = (datum_from_line(l) for l in input_file)
        for datum in data.transform(raw_data, feature_params):
            example = data.datum_to_tf_example(datum)
            writer.write(example.SerializeToString())

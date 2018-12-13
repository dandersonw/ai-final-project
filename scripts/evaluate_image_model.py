import tensorflow as tf

import argparse
import importlib
import sys

MY_SRC = '../src/'
sys.path.append(MY_SRC)

from image_model import densenet121_model
import data
from data import IMAGE_DIMS, NUM_CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()

    features = {'image'}

    with tf.Session() as sess:
        testing_data = data.preload_dataset(args.test, sess, features=features)

    model = densenet121_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(clipnorm=5.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.load_weights(args.checkpoint_path)
    scores = model.evaluate(x=testing_data[0]['image'], y=testing_data[1])
    for (metric, score) in zip(model.metrics_names, scores):
        print('{}: {:.4f}'.format(metric, score))


if __name__ == '__main__':
    main()

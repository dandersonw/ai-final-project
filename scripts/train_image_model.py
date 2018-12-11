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
    parser.add_argument('--train', required=True)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()

    features = {'image'}
    with tf.Session() as sess:
        training_data = data.preload_dataset(args.train, sess, features=features)
        validation_data = data.preload_dataset(args.valid, sess, features=features)
        testing_data = data.preload_dataset(args.test, sess, features=features)

    model = train_model(training_data, validation_data, args.checkpoint_path)
    model.evaluate(x=testing_data[0]['image'], y=testing_data[1])


def train_model(training_data, validation_data, checkpoint_path):
    model = densenet121_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=NUM_CLASSES)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    compilation_kwargs = {'optimizer': tf.keras.optimizers.Adam(clipnorm=5.0),
                          'loss': 'categorical_crossentropy',
                          'metrics': ['accuracy']}
    model.compile(**compilation_kwargs)

    train_kwargs = {'x': training_data[0]['image'],
                    'y': training_data[1],
                    'validation_data': (validation_data[0]['image'], validation_data[1]),
                    'callbacks': [checkpoint],
                    'verbose': 1}
                   # 'steps_per_epoch': 1,
                   # 'validation_steps': 1}

    model.fit(**train_kwargs, epochs=30)
    #model.save_weights(checkpoint_path)
    model.load_weights(checkpoint_path)
    
    return model

if __name__ == '__main__':
    main()

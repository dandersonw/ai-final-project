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
    parser.add_argument('--imagenet', default=False)
    parser.add_argument('--checkpoint_path', required=True)
    args = parser.parse_args()
    
    features = {'image'}
    with tf.Session() as sess:
        training_data = data.preload_dataset(args.train, sess, features=features)
        validation_data = data.preload_dataset(args.valid, sess, features=features)
        
    model = train_model(training_data, validation_data, args.checkpoint_path, args.imagenet)


def train_model(training_data, validation_data, checkpoint_path, use_imagenet):
    model = densenet121_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=NUM_CLASSES, use_trained_weights=use_imagenet)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10)
    compilation_kwargs = {'optimizer': tf.keras.optimizers.Adam(clipnorm=5.0),
                          'loss': 'categorical_crossentropy',
                          'metrics': ['accuracy']}
    model.compile(**compilation_kwargs)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-5)

    train_kwargs = {'x': training_data[0]['image'],
                    'y': training_data[1],
                    'validation_data': (validation_data[0]['image'], validation_data[1]),
                    'callbacks': [checkpoint, early_stopping, reduce_lr],
                    'verbose': 1}

    model.fit(**train_kwargs, epochs=30)

    model.load_weights(checkpoint_path)
    
    return model

if __name__ == '__main__':
    main()

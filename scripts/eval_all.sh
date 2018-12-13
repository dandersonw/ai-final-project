#!/bin/bash
AI_RESOURCE_PATH='../data/resources/' python evaluate_image_model.py --test ../data/tfrecord/modern_testing_data.tfrecord --checkpoint_path ./checkpoints/$1
AI_RESOURCE_PATH='../data/resources/' python evaluate_image_model.py --test ../data/tfrecord/all_testing_data.tfrecord --checkpoint_path ./checkpoints/$1
AI_RESOURCE_PATH='../data/resources/' python evaluate_image_model.py --test ../data/tfrecord/legacy_testing_data.tfrecord --checkpoint_path ./checkpoints/$1

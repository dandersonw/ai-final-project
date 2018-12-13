#!/bin/bash
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/all_training_data.tfrecord --valid ../data/tfrecord/all_validation_data.tfrecord --imagenet True --checkpoint_path ./checkpoints/all-adam-1
sleep 60
./eval_all.sh all-adam-1 > all-adam.txt
sleep 60
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/modern_training_data.tfrecord --valid ../data/tfrecord/modern_validation_data.tfrecord --imagenet True --checkpoint_path ./checkpoints/modern-adam-1
sleep 60
./eval_all.sh modern-adam-1 > modern-adam-1.txt
sleep 60
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/legacy_training_data.tfrecord --valid ../data/tfrecord/legacy_validation_data.tfrecord --imagenet True --checkpoint_path ./checkpoints/legacy-adam-1
sleep 60
./eval_all.sh legacy-adam-1 > legacy-adam-1.txt
sleep 60
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/all_training_data.tfrecord --valid ../data/tfrecord/all_validation_data.tfrecord --checkpoint_path ./checkpoints/all-raw-adam
sleep 60
./eval_all.sh all-raw-adam > all-raw-adam.txt
sleep 60
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/modern_training_data.tfrecord --valid ../data/tfrecord/modern_validation_data.tfrecord --checkpoint_path ./checkpoints/modern-raw-adam
sleep 60
./eval_all.sh modern-raw-adam > modern-raw-adam.txt
sleep 60
AI_RESOURCE_PATH='../data/resources/' python train_image_model.py --train ../data/tfrecord/legacy_training_data.tfrecord --valid ../data/tfrecord/legacy_validation_data.tfrecord --checkpoint_path ./checkpoints/legacy-raw-adam
sleep 60
./eval_all.sh legacy-raw-adam > legacy-raw-adam.txt

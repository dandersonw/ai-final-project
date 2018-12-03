import tensorflow as tf
import numpy as np
import pymysql.cursors
import sys
sys.path.append("../src")
import data
import json

from dataTransform import transform_data

db = pymysql.connect(host="localhost",
                     user="root",
                     passwd="root",
                     db="aiproject",
                     port=3307)

def convert_data(output_path, status):
    with db.cursor() as cursor:
        sql = "select * from unique_card_data where experiment_status = %s"
        cursor.execute(sql, status)
        responses = cursor.fetchall()
        #print(responses[0])
        writer = tf.python_io.TFRecordWriter(output_path)
        raw_data = (transform_data(r) for r in responses)
        data_size = len(responses)
        counter = 1
        for datum in raw_data:
            example = data.datum_to_tf_example(datum)
            writer.write(example.SerializeToString())
            print("wrote", counter, "of", data_size)
            counter += 1

#convert_data("./testing_data.tfrecord", "TESTING")
convert_data("./training_data.tfrecord", "TRAINING")
convert_data("./validation_data.tfrecord", "VALIDATION")

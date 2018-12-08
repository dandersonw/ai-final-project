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

def convert_data(output_path, status, setSql):
    with db.cursor() as cursor:
        sql = "select * from unique_card_data where experiment_status = %s " + setSql
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

#convert_data("./tfrecord/testing_data.tfrecord", "TESTING")
#convert_data("./tfrecord/training_data.tfrecord", "TRAINING")
#convert_data("./tfrecord/validation_data.tfrecord", "VALIDATION")

modern_sql = "and set_name in (select setCode from sets where releaseDate >= date(\"2008-10-03\"))"
convert_data("./tfrecord/modern_training_data.tfrecord", "TRAINING", modern_sql)
convert_data("./tfrecord/modern_validation_data.tfrecord", "VALIDATION", modern_sql)
convert_data("./tfrecord/modern_testing_data.tfrecord", "TESTING", modern_sql)

legacy_sql = "and set_name in (select setCode from sets where releaseDate < date(\"2008-10-03\"))"
convert_data("./tfrecord/legacy_training_data.tfrecord", "TRAINING", legacy_sql)
convert_data("./tfrecord/legacy_validation_data.tfrecord", "VALIDATION", legacy_sql)
convert_data("./tfrecord/legacy_testing_data.tfrecord", "TESTING", legacy_sql)

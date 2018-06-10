import tensorflow as tf
from datasets.dataset_utils import create_example
from datasets import create_squad_records

if __name__ == "__main__":
    # Create test squad data records.
    squad_tuples = [
        ("Obama was president.", "Who was Obama?",
         "president", 1, 2, 3),
        ("He was popular.", "", "", 0, -1, -1),
        ("Washington DC was where he was.", "Where was Obama?",
         "Washington DC", 1, 0, 2),
        ("He married Michelle in 1920.", "Who married Obama?",
         "Michelle", 1, 2, 3),
        ("He hated bad people.", "", "", 0, -1, -1),
    ]
    tfrecord_writer = tf.python_io.TFRecordWriter(
        "datasets/testdata/squad_test.tfrecords")
    for tup in squad_tuples:
        tf_example = create_example(tup, create_squad_records.FEATURE_NAMES)
        tfrecord_writer.write(tf_example.SerializeToString())
    tfrecord_writer.close()

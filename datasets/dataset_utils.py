"""Utilities for data related tasks."""

import re
import string
import tensorflow as tf


def int64_feature(value):
    """Creates feature for integers."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """Creates feature for bytes/strings."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_list_feature(value):
    """Creates feature for float lists."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example(values, names):
    """Creates an example from given values."""
    tf_features = {}
    for i, val in enumerate(values):
        if isinstance(val, int):
            tf_features[names[i]] = int64_feature(val)
        elif isinstance(val, str):
            tf_features[names[i]] = bytes_feature(val.encode("utf-8"))
        elif isinstance(val, bytes):
            tf_features[names[i]] = bytes_feature(val)
    return tf.train.Example(features=tf.train.Features(feature=tf_features))


def strip_punctuation(s):
    """Strips punctuation from text."""
    return re.sub("[" + string.punctuation + "]", "", s)


def write_to_file(examples, filepath):
    """Writes serialized examples to filepath"""
    tfrecord_writer = tf.python_io.TFRecordWriter(filepath)
    for ex in examples:
        tfrecord_writer.write(ex.SerializeToString())
    tfrecord_writer.close()

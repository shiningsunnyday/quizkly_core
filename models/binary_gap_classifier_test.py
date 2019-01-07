"""Test methods fopr creating QuesitonCandidate Protos."""

import os
import unittest

import tensorflow as tf

from models.binary_gap_classifier import train_model


class TestBinaryGapClassifier(unittest.TestCase):
    """ Tests for binary gap classifier. """

    def test_train_model(self):
        hparams = tf.contrib.training.HParams(
            learning_rate=0.9, hidden_size=100, encoding_dim=3
        )
        output_file = "datasets/testdata/gap_classifier_test.h5"
        train_model(
            hparams, "datasets/testdata/question_candidates_test", output_file
        )
        os.remove(output_file)


if __name__ == "__main__":
    unittest.main()

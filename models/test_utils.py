"""Class for testing models."""

import os
import shutil
import tempfile

import tensorflow as tf

from models.base_model import Mode
from trainer.training import train_sess
from trainer.evaluation import evaluate_sess


class ModelTester(tf.test.TestCase):

    def setUp(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self._test_dir = tempfile.mkdtemp()
        super(ModelTester, self).setUp()

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def get_model(self, mode):
        raise NotImplementedError("Sub-classes should implement this method.")

    def test_training(self):
        model = self.get_model(mode=Mode.TRAIN)
        with tf.Session() as sess:
            model.build_model()
            writer = tf.summary.FileWriter(
                os.path.join(self._test_dir, 'train_summaries'),
                sess.graph)
            sess.run(model.variable_init_op)
            train_sess(sess, model, num_steps=10, writer=writer)

    def test_eval(self):
        model = self.get_model(mode=Mode.EVAL)
        with tf.Session() as sess:
            model.build_model()
            writer = tf.summary.FileWriter(
                os.path.join(self._test_dir, 'eval_summaries'),
                sess.graph)
            sess.run([model.metrics_init_op, model.variable_init_op])
            evaluate_sess(sess, model, num_batches=3, writer=writer)

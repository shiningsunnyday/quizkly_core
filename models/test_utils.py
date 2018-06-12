"""Class for testing models."""

import shutil
import tempfile

import tensorflow as tf

from models.run_training import train_eval_model


class ModelTester(tf.test.TestCase):
    def setUp(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self._test_dir = tempfile.mkdtemp()
        super(ModelTester, self).setUp()

    def tearDown(self):
        shutil.rmtree(self._test_dir)

    def get_model(self):
        raise NotImplementedError("Sub-classes should implement this method.")

    def get_hparams(self):
        raise NotImplementedError("Sub-classes should implement this method.")

    def test_training_eval(self):
        hparams = self.get_hparams()
        model = self.get_model()
        run_config = tf.estimator.RunConfig()
        run_config = run_config.replace(
            model_dir=self._test_dir,
            save_checkpoints_steps=2,
            save_summary_steps=2,
        )
        train_eval_model(
            hparams,
            model.model_fn,
            model.input_fn,
            run_config,
            train_steps=5,
            eval_steps=2,
        )

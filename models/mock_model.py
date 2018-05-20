"""A mock model for testing purposes."""

import tensorflow as tf

from models.base_model import BaseModel, Mode

_DATA = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
_LABELS = [1.0, 0.0, 1.0]


class HParams(tf.contrib.training.HParams):
    learning_rate = None


class MockModel(BaseModel):

    def build_model(self):
        """Build tf graph."""
        self.global_step = tf.train.get_or_create_global_step()
        weights = tf.get_variable("weights", shape=[2, 1])
        out = tf.squeeze(
            tf.matmul(tf.constant(_DATA), weights))
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.constant(_LABELS),
                logits=out)
        )
        tf.summary.scalar("loss", self.loss)

        if self._mode == Mode.TRAIN:
            opt = tf.train.AdagradOptimizer(
                learning_rate=self._hparams.learning_rate)
            self.train_op = opt.minimize(
                self.loss, self.global_step)
        elif self._mode == Mode.EVAL:
            self.metrics = {}
            self.metrics["accuracy"] = tf.metrics.accuracy(
                labels=tf.constant(_LABELS),
                predictions=out > 0.5)
            metric_variables = tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES)
            self.metrics_init_op = tf.variables_initializer(
                metric_variables)
        self.variable_init_op = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()

    def export(self, builder, session):
        assert self._mode == Mode.INFERENCE
        builder.add_meta_graph_and_variables(session, ["inference"])


HPARAMS = HParams(learning_rate=0.01)

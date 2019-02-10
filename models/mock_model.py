"""A mock model for testing purposes."""

import tensorflow as tf

_DATA = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
_LABELS = [1.0, 0.0, 1.0]

FEATURE_NAME = "feature"
LABEL_NAME = "label"

class HParams(tf.contrib.training.HParams):
    learning_rate = None
    batch_size = None


def input_fn(hparams, mode):
    """Load and return dataset of batched examples.

    Args:
        hparams: hyperparameters that the fn can depend on.
        mode: can process data differently for training, eval or
            inference according to mode.
    """

    def _input_fn_callable():
        dataset = tf.data.Dataset.from_tensor_slices((_DATA, _LABELS))
        dataset = dataset.map(_parse_fn)
        dataset = dataset.shuffle(buffer_size=500 * hparams.batch_size)
        dataset = dataset.repeat().batch(hparams.batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def _parse_fn(feature, labels):
        parsed = {}
        parsed[FEATURE_NAME] = feature
        parsed[LABEL_NAME] = labels
        return parsed, parsed[LABEL_NAME]

    def serving_input_receiver_fn():
        """Function for exported serving model"""
        inputs = {
            FEATURE_NAME: tf.placeholder(shape=[None, 2], dtype=tf.float32)
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return serving_input_receiver_fn
    return _input_fn_callable


def model_fn(features, labels, mode, params):
    """ Defines how to train, evaluate and predict from model.
    Refer to: https://www.tensorflow.org/get_started/premade_estimators
    """
    loss, train_op, metrics, predictions = [None] * 4
    weights = tf.get_variable("weights", shape=[2, 1])
    out = tf.squeeze(tf.matmul(features[FEATURE_NAME], weights))
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=out)
        )
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdagradOptimizer(learning_rate=params.learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    else:
        predictions = out >= 0.5
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}
        metrics["accuracy"] = tf.metrics.accuracy(
            labels=labels, predictions=predictions
        )
        metrics["mean_loss"] = tf.metrics.mean(loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
    )


HPARAMS = HParams(learning_rate=0.01, batch_size=2)

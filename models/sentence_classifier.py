"""Model to classify sentences."""
import tensorflow as tf
import tensorflow_hub as hub

# Constants to be used for saving/serving.
SENTENCE_SIGNATURE = "sentence"
PREDICTION_SIGNATURE = "predictions"


class HParams(tf.contrib.training.HParams):
    train_records = None
    eval_records = None
    sentence_feature = None
    label_feature = None
    train_batch_size = None
    eval_batch_size = None
    hidden_size = None
    num_classes = None
    compression_type = None
    is_test = None


def input_fn(params, mode):
    """Load and return dataset of batched examples.

     Args:
         params: hyperparameters that the fn can depend on.
         mode: can process data differently for training, eval or
             inference according to mode.
    """

    def serving_input_receiver_fn():
        """Function for exported serving model"""
        inputs = {
            SENTENCE_SIGNATURE: tf.placeholder(shape=[None], dtype=tf.string)
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return serving_input_receiver_fn

    feature_spec = {
        params.sentence_feature: tf.FixedLenFeature([], tf.string),
        params.label_feature: tf.FixedLenFeature([], tf.int64),
    }

    def _parse_fn(example_proto):
        parsed = tf.parse_single_example(example_proto, feature_spec)
        parsed[SENTENCE_SIGNATURE] = parsed.pop(params.sentence_feature)
        return parsed, parsed[params.label_feature]

    def input_fn_callable():
        batch_size = params.eval_batch_size
        file_patterns = params.eval_records
        if mode == tf.estimator.ModeKeys.TRAIN:
            batch_size = params.train_batch_size
            file_patterns = params.train_records
        dataset = tf.data.TFRecordDataset(
            tf.gfile.Glob(file_patterns),
            compression_type=params.compression_type,
        )
        dataset = dataset.map(_parse_fn)
        dataset = dataset.shuffle(buffer_size=500 * batch_size)
        dataset = dataset.repeat().batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn_callable


def model_fn(features, labels, mode, params):
    """ Defines how to train, evaluate and predict from model.
    Refer to: https://www.tensorflow.org/get_started/premade_estimators
    """
    loss, train_op, metrics, predictions, export_outputs = [None] * 5
    if params.is_test:
        embedded_sentences = tf.random_uniform(
            [tf.shape(features[SENTENCE_SIGNATURE])[0], 256]
        )
    else:
        sentence_encoder = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder/2"
        )
        sentences = features[SENTENCE_SIGNATURE]
        embedded_sentences = sentence_encoder(sentences)
    tf.summary.histogram("embedded", embedded_sentences)
    embedded_sentences = tf.layers.dense(
        embedded_sentences, params.hidden_size, activation=tf.tanh
    )
    logits = tf.layers.dense(embedded_sentences, params.num_classes)
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = tf.reduce_mean(loss, axis=0)
        tf.summary.scalar("loss", loss)
    if mode != tf.estimator.ModeKeys.TRAIN:
        predictions = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(learning_rate=0.0005)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}
        metrics["accuracy"] = tf.metrics.accuracy(
            labels=labels, predictions=predictions
        )
        metrics["majority"] = tf.metrics.mean(values=labels)
        metrics["recall"] = tf.metrics.recall(
            labels=labels, predictions=predictions
        )
        metrics["precision"] = tf.metrics.precision(
            labels=labels, predictions=predictions
        )
        metrics["average"] = tf.metrics.mean(values=predictions)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            PREDICTION_SIGNATURE: tf.estimator.export.PredictOutput(
                {PREDICTION_SIGNATURE: predictions}
            )
        }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs=export_outputs,
    )

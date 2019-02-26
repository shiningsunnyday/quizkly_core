"""Model to rank distractors."""
import tensorflow as tf
import tensorflow_hub as hub

# Constants to be used for saving/serving.
QUESTION_SIGNATURE = "question"
ANSWER_SIGNATURE = "answer"
ENCODING_SIGNATURE = "encoding"
PREDICTION_SIGNATURE = "predictions"


class HParams(tf.contrib.training.HParams):
    train_records = None
    eval_records = None
    question_feature = None
    answer_feature = None
    distractor_feature = None
    negatives_feature = None
    negatives_length_feature = None
    vector_size = None
    train_batch_size = None
    eval_batch_size = None
    compression_type = None
    is_test = None
    learning_rate = None


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
            QUESTION_SIGNATURE: tf.placeholder(shape=[None], dtype=tf.string),
            ANSWER_SIGNATURE: tf.placeholder(shape=[None], dtype=tf.string),
        }
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return serving_input_receiver_fn

    context_feature_spec = {
        params.question_feature: tf.FixedLenFeature([], tf.string),
        params.answer_feature: tf.FixedLenFeature([], tf.string),
        params.distractor_feature: tf.FixedLenFeature(
            [params.vector_size], tf.float32),
        params.negatives_length_feature: tf.FixedLenFeature([], tf.int64)
    }
    sequence_feature_spec = {
        params.negatives_feature: tf.FixedLenSequenceFeature(
            [params.vector_size], dtype=tf.float32)
    }

    def _parse_fn(example_proto):
        features, sequence_features = tf.parse_single_sequence_example(
            example_proto, context_feature_spec, sequence_feature_spec)
        parsed = {**features, **sequence_features}
        parsed[QUESTION_SIGNATURE] = parsed.pop(params.question_feature)
        parsed[ANSWER_SIGNATURE] = parsed.pop(params.answer_feature)
        return parsed

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
        embedded_questions = tf.random_uniform(
            [tf.shape(features[QUESTION_SIGNATURE])[0], 256]
        )
        embedded_answers = tf.random_uniform(
            [tf.shape(features[ANSWER_SIGNATURE])[0], 256]
        )
    else:
        sentence_encoder = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder/2"
        )
        questions = features[QUESTION_SIGNATURE]
        embedded_questions = sentence_encoder(questions)
        answers = features[ANSWER_SIGNATURE]
        embedded_answers = sentence_encoder(answers)
    tf.summary.histogram("embedded_questions", embedded_questions)
    tf.summary.histogram("embedded_answers", embedded_answers)
    query_encoding = tf.layers.dense(
        tf.concat([embedded_questions, embedded_answers], axis=1),
        params.vector_size, activation=tf.tanh, use_bias=False)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = query_encoding
        export_outputs = {
            PREDICTION_SIGNATURE: tf.estimator.export.PredictOutput(
                {ENCODING_SIGNATURE: query_encoding}
            )
        }
    if mode != tf.estimator.ModeKeys.PREDICT:
        batch_size = params.eval_batch_size
        if mode == tf.estimator.ModeKeys.TRAIN:
            batch_size = params.train_batch_size
        keys = tf.concat(
            [
                tf.expand_dims(features[params.distractor_feature], axis=1),
                features[params.negatives_feature]
            ],
            axis=1)
        logits = tf.squeeze(
            tf.matmul(keys, tf.expand_dims(query_encoding, axis=2)),
            axis=2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([batch_size], dtype=tf.int64), logits=logits
        )
        loss = tf.reduce_mean(loss, axis=0)
        tf.summary.scalar("loss", loss)
    if mode == tf.estimator.ModeKeys.EVAL:
        predictions = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}
        metrics["accuracy"] = tf.metrics.accuracy(
            labels=tf.zeros([batch_size], dtype=tf.int64),
            predictions=predictions
        )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs=export_outputs,
    )

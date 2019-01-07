"""Model to classify sentences."""
import tensorflow as tf
import tensorflow_hub as hub


# Constants to be used for saving/serving.
SENTENCE_SIGNATURE = "sentence"
START_PREDICTION_SIGNATURE = "start"
END_PREDICTION_SIGNATURE = "end"


class HParams(tf.contrib.training.HParams):
    train_records = None
    eval_records = None
    sentence_feature = None
    start_label_feature = None
    end_label_feature = None
    sentence_length_feature = None
    train_batch_size = None
    eval_batch_size = None
    hidden_size = None
    dropout_keep_prob = None
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
        params.start_label_feature: tf.FixedLenFeature([], tf.int64),
        params.end_label_feature: tf.FixedLenFeature([], tf.int64),
        params.sentence_length_feature: tf.FixedLenFeature([], tf.int64),
    }

    def _parse_fn(example_proto):
        parsed = tf.parse_single_example(example_proto, feature_spec)
        parsed[SENTENCE_SIGNATURE] = parsed.pop(params.sentence_feature)
        return parsed, parsed[params.start_label_feature]

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
        dataset = dataset.filter(
            lambda features, label: features[params.start_label_feature] >= 0
        )
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
        sentences_shape = tf.shape(features[SENTENCE_SIGNATURE])
        sequence_embeddings = tf.random_uniform([sentences_shape[0], 10, 256])
    else:
        elmo = hub.Module("https://tfhub.dev/google/elmo/2")
        sentences = features[SENTENCE_SIGNATURE]
        tf.summary.text("sentences", sentences)
        sparse_words = tf.string_split(sentences, skip_empty=False)
        words = tf.sparse_tensor_to_dense(sparse_words, default_value="<UNK>")
        # Mask the empty strings added in sparse to dense.
        # Mask padded embeddings.
        sequence_lengths = features[params.sentence_length_feature]
        sequence_embeddings = elmo(
            inputs={
                "tokens": words,
                "sequence_len": tf.to_int32(sequence_lengths),
            },
            signature="tokens",
            as_dict=True,
        )["elmo"]
        # Mask the empty strings added in sparse to dense.
        # Mask padded embeddings.
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        sequence_embeddings = tf.multiply(
            sequence_embeddings, tf.expand_dims(mask, -1)
        )

    tf.summary.histogram("embedded", sequence_embeddings)
    # RNN.
    cell_fw = tf.contrib.rnn.GRUCell(num_units=params.hidden_size)
    cell_bw = tf.contrib.rnn.GRUCell(num_units=params.hidden_size)

    if mode == tf.estimator.ModeKeys.TRAIN:
        keep_prob = params.dropout_keep_prob
        cell_fw = tf.contrib.rnn.DropoutWrapper(
            cell_fw, output_keep_prob=keep_prob, state_keep_prob=keep_prob
        )
        cell_bw = tf.contrib.rnn.DropoutWrapper(
            cell_bw, output_keep_prob=keep_prob, state_keep_prob=keep_prob
        )

    (output_fw, output_bw), (
        state_fw,
        state_bw,
    ) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=sequence_embeddings,
        dtype=tf.float32,
        sequence_length=features.get(params.sentence_length_feature, None),
    )
    outputs = (output_fw + output_bw) / 2.0

    # Pointer RNN.
    decoder_cell = tf.contrib.rnn.GRUCell(num_units=params.hidden_size)
    state = tf.zeros([tf.shape(outputs)[0], params.hidden_size])
    initial_input = (state_fw + state_bw) / 2.0
    state, start_output = decoder_cell(initial_input, state)

    start_pointer_logits = _attention_logits(
        start_output, outputs, params.hidden_size, name="start"
    )
    start_pointer_scores = tf.expand_dims(
        tf.nn.softmax(start_pointer_logits), axis=2
    )
    if mode != tf.estimator.ModeKeys.TRAIN:
        end_input = tf.reduce_sum(start_pointer_scores * outputs)
    else:
        batch_ids = tf.range(params.train_batch_size, dtype=tf.int64)
        start_labels = features[params.start_label_feature]
        gather_ids = tf.stack((batch_ids, start_labels), axis=1)
        end_input = tf.gather_nd(outputs, gather_ids)

    _, end_output = decoder_cell(end_input, state)
    end_pointer_logits = _attention_logits(
        end_output, outputs, params.hidden_size, name="end"
    )

    if mode != tf.estimator.ModeKeys.PREDICT:
        start_labels = features[params.start_label_feature]
        end_labels = features[params.end_label_feature]
        tf.summary.text("start_labels", tf.as_string(start_labels))
        tf.summary.text("end_labels", tf.as_string(end_labels))
        start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=start_labels, logits=start_pointer_logits
        )
        start_loss = tf.reduce_mean(start_loss, axis=0)
        end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=end_labels, logits=end_pointer_logits
        )
        end_loss = tf.reduce_mean(end_loss, axis=0)
        loss = start_loss + end_loss
        tf.summary.scalar("aggregate/loss", loss)
        tf.summary.scalar("start/loss", start_loss)
        tf.summary.scalar("end/loss", end_loss)
    if mode != tf.estimator.ModeKeys.TRAIN:
        start_predictions = tf.argmax(start_pointer_logits, axis=1)
        end_predictions = tf.argmax(end_pointer_logits, axis=1)
        tf.summary.text("start_predictions", tf.as_string(start_predictions))
        tf.summary.text("end_predictions", tf.as_string(end_predictions))
        predictions = {
            START_PREDICTION_SIGNATURE: start_predictions,
            END_PREDICTION_SIGNATURE: end_predictions,
        }
    if mode == tf.estimator.ModeKeys.TRAIN:
        opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:
        metrics = {}
        metrics["start/accuracy"] = tf.metrics.accuracy(
            labels=start_labels, predictions=start_predictions
        )
        metrics["end/accuracy"] = tf.metrics.accuracy(
            labels=end_labels, predictions=end_predictions
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "default": tf.estimator.export.PredictOutput(
                {
                    START_PREDICTION_SIGNATURE: start_predictions,
                    END_PREDICTION_SIGNATURE: end_predictions,
                }
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


def _attention_logits(query_vector, memory, num_units, name="attn"):
    query_attn_layer = tf.layers.Dense(
        num_units, use_bias=False, name=name + "_query_layer"
    )
    mem_attn_layer = tf.layers.Dense(
        num_units, use_bias=False, name=name + "_mem_layer"
    )
    v = tf.get_variable(name + "_v", [num_units], dtype=tf.float32)
    query_proj = query_attn_layer(query_vector)
    query_proj = tf.expand_dims(query_proj, axis=1)
    memory_proj = mem_attn_layer(memory)
    return tf.reduce_sum(v * tf.tanh(query_proj + memory_proj), axis=2)

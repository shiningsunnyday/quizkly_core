"""Model to classify sentences."""
import tensorflow as tf
import tensorflow_hub as hub

from models.base_model import BaseModel, Mode

PREDICTION_SIGNATURE = "prediction"
SENTENCE_SIGNATURE = "sentence"

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

class SentenceClassifier(BaseModel):

    def build_model(self):
        """Build graph."""
        self.global_step = tf.train.get_or_create_global_step()
        if self._mode != Mode.INFERENCE:
            sentences, labels = self._read_data()
            tf.summary.text("input_sentences", sentences)
        else:
            sentences = tf.placeholder(
                dtype=tf.string, shape=[None],
                name=SENTENCE_SIGNATURE)
        sentence_encoder = self._hub_module
        embedded_sentences = sentence_encoder(sentences)
        tf.summary.histogram("embedded", embedded_sentences)
        # embedded_sentences = tf.layers.dropout(
        #     embedded_sentences, training=self._mode==Mode.TRAIN) 
        embedded_sentences = tf.layers.dense(
            embedded_sentences, self._hparams.hidden_size,
            activation=tf.tanh
        )
        logits = tf.layers.dense(
            embedded_sentences, self._hparams.num_classes
        )
        if self._mode != Mode.INFERENCE:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            self.loss = tf.reduce_mean(self.loss, axis=0)
            tf.summary.scalar("loss", self.loss)
        if self._mode != Mode.TRAIN:
            self._predictions = tf.argmax(logits, axis=1)
            tf.identity(self._predictions, name=PREDICTION_SIGNATURE)
        if self._mode == Mode.TRAIN:
            opt = tf.train.AdamOptimizer(
                learning_rate=0.0005)
            self.train_op = opt.minimize(
                self.loss, self.global_step)
        elif self._mode == Mode.EVAL:
            self.metrics = {}
            self.metrics["accuracy"] = tf.metrics.accuracy(
                labels=labels,
                predictions=self._predictions)
            self.metrics["majority"] = tf.metrics.mean(
                values=labels
            )
            self.metrics["recall"] = tf.metrics.recall(
                labels=labels,
                predictions=self._predictions)
            self.metrics["precision"] = tf.metrics.precision(
                labels=labels,
                predictions=self._predictions)
            self.metrics["average"] = tf.metrics.mean(
                values=self._predictions
            )
            metric_variables = tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES)
            self.metrics_init_op = tf.variables_initializer(
                metric_variables)
        self.variable_init_op = [tf.global_variables_initializer(),
                                 tf.tables_initializer()]
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def _read_data(self):
        feature_spec = {
            self._hparams.sentence_feature: tf.FixedLenFeature(
                [], tf.string),
            self._hparams.label_feature: tf.FixedLenFeature(
                [], tf.int64
            )
        }
        def _parse_fn(example_proto):
            parsed = tf.parse_single_example(example_proto, feature_spec)
            return [parsed[self._hparams.sentence_feature],
                    parsed[self._hparams.label_feature]]
        batch_size = self._hparams.eval_batch_size
        file_patterns = self._hparams.eval_records
        if self._mode == Mode.TRAIN:
            batch_size = self._hparams.train_batch_size
            file_patterns = self._hparams.train_records
        dataset = tf.data.TFRecordDataset(
            tf.gfile.Glob(file_patterns),
            compression_type=self._hparams.compression_type)
        dataset = dataset.map(_parse_fn)
        dataset = dataset.shuffle(buffer_size=500 * batch_size)
        dataset = dataset.repeat().batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    @classmethod
    def export_model(cls, hparams, checkpoint, builder):
        with tf.Session(graph=tf.Graph()) as sess:
            hub_module = hub.Module(
                "https://tfhub.dev/google/universal-sentence-encoder/1")
            inference_model = cls(hparams, Mode.INFERENCE, hub_module)
            inference_model.build_model()
            inference_model.saver.restore(sess, checkpoint)
            sess.run(tf.tables_initializer())
            builder.add_meta_graph_and_variables(
                sess, ["inference"],
                legacy_init_op=tf.tables_initializer())
            builder.save()
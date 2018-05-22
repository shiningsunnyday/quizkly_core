"""Model to classify sentences."""
import tensorflow as tf
import tensorflow_hub as hub

from models.base_model import BaseModel, Mode

class HParams(tf.contrib.training.HParams):
    train_records = None
    eval_records = None
    sentence_feature = None
    label_feature = None
    train_batch_size = None
    eval_batch_size = None
    hidden_size = None
    num_classes = None

class SentenceClassifier(BaseModel):

	def build_model():
		"""Build graph."""
		self.global_step = tf.train.get_or_create_global_step()
		sentences, labels = self._read_data()
		sentence_encoder = hub.Module(
			"https://tfhub.dev/google/universal-sentence-encoder/1")
		embedded_sentences = sentence_encoder(sentences)
        embedded_sentences = tf.layers.dense(
        	embedded_sentences, self._hparams.hidden_size,
        	activation=tf.tanh
        )
        logits = tf.layers.dense(
        	embedded_sentences, num_classes
        )
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        	labels=labels, logits=logits)
        tf.summary.scalar("loss", self.loss)
        if self._mode != Mode.TRAIN:
        	self._predictions = tf.argmax(logits)
        if self._mode == Mode.TRAIN:
            opt = tf.train.AdagradOptimizer(
                learning_rate=self._hparams.learning_rate)
            self.train_op = opt.minimize(
                self.loss, self.global_step)
        elif self._mode == Mode.EVAL:
            self.metrics = {}
            self.metrics["accuracy"] = tf.metrics.accuracy(
                labels=labels,
                predictions=self._predictions)
            metric_variables = tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES)
            self.metrics_init_op = tf.variables_initializer(
                metric_variables)
        self.variable_init_op = tf.global_variables_initializer()
        self.summary_op = tf.summary.merge_all()

	def _read_data():
        feature_spec = {
		    self._hparams.sentence_feature: tf.FixedLenFeature(
                [], tf.string)
		    self._hparams.label_feature: tf.FixedLenFeature(
		    	[], tf.int32
		    )
		}
		def _parse_fn(example_proto):
			parsed = tf.parse_single_example(example_proto, feature_spec)
			return [parsed[self._hparams.sentence_feature],
			        self._hparams.label_feature]
		batch_size = self._hparams.eval_batch_size
		file_patterns = self._hparam.eval_records
		if self._mode == Mode.TRAIN:
			batch_size = self._hparams.train_batch_size
			file_patterns = self._hparams.train_records
		dataset = tf.data.TFRecordDataset(
			tf.gfile.Glob(file_patterns))
		dataset = dataset.map(_parse_fn)
		dataset = dataset.shuffle(buffer_size=50 * batch_size)
		dataset = dataset.repeat().batch(batch_size)
		return dataset.make_one_shot_iterator().get_next()

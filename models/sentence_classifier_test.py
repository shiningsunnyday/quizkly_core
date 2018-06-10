import tensorflow as tf
import tensorflow_hub as hub

from models import test_utils
from models.sentence_classifier import SentenceClassifier
from models.sentence_classifier import HParams


class TestSentenceClassifier(test_utils.ModelTester):
    def get_model(self, mode):
        hub_module = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder/1")
        hparams = HParams(
            train_records="datasets/testdata/*.tfrecords",
            eval_records="datasets/testdata/*.tfrecords",
            sentence_feature="sentence",
            label_feature="question_worthy",
            train_batch_size=3,
            eval_batch_size=3,
            hidden_size=5,
            num_classes=2
        )
        return SentenceClassifier(
            mode=mode, hparams=hparams, hub_module=hub_module)


if __name__ == "__main__":
    tf.test.main()

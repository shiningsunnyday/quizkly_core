import tensorflow as tf

from models import test_utils
from models import sentence_classifier
from models.sentence_classifier import HParams


class TestSentenceClassifier(test_utils.ModelTester):
    def get_model(self):
        return sentence_classifier

    def get_hparams(self):
        return HParams(
            train_records="datasets/testdata/*.tfrecords",
            eval_records="datasets/testdata/*.tfrecords",
            sentence_feature="sentence",
            context_feature="context",
            label_feature="question_worthy",
            train_batch_size=3,
            eval_batch_size=3,
            hidden_size=5,
            num_classes=2,
            learning_rate=0.005,
            is_test=True,
        )


if __name__ == "__main__":
    tf.test.main()

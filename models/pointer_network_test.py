import tensorflow as tf

from models import test_utils
from models import pointer_network
from models.pointer_network import HParams


class TestPointerNetwork(test_utils.ModelTester):
    def get_model(self):
        return pointer_network

    def get_hparams(self):
        return HParams(
            train_records="datasets/testdata/*.tfrecords",
            eval_records="datasets/testdata/*.tfrecords",
            sentence_feature="sentence",
            start_label_feature="answer_start",
            end_label_feature="answer_end",
            sentence_length_feature="sentence_length",
            train_batch_size=3,
            eval_batch_size=3,
            hidden_size=5,
            dropout_keep_prob=0.5,
            learning_rate=0.0005,
            is_test=True,
        )


if __name__ == "__main__":
    tf.test.main()

import tensorflow as tf

from models import test_utils
from models import distractor_encoder
from models.distractor_encoder import HParams


class TestDistractorEncoder(test_utils.ModelTester):
    def get_model(self):
        return distractor_encoder

    def get_hparams(self):
        return HParams(
            train_records="datasets/testdata/distractors_test",
            eval_records="datasets/testdata/distractors_test",
            question_feature="question",
            answer_feature="answer",
            distractor_feature="distractor",
            negatives_feature="negatives",
            negatives_length_feature="negatives_length",
            vector_size=10,
            train_batch_size=5,
            eval_batch_size=5,
            learning_rate=0.001,
            is_test=True,
        )


if __name__ == "__main__":
    tf.test.main()

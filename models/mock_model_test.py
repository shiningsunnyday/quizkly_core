import tensorflow as tf

from models import test_utils
from models import mock_model
from models.mock_model import HPARAMS


class TestMockModel(test_utils.ModelTester):
    def get_model(self):
        return mock_model

    def get_hparams(self):
        return HPARAMS


if __name__ == "__main__":
    tf.test.main()

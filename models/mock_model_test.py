import tensorflow as tf

from models import test_utils
from models.mock_model import MockModel
from models.mock_model import HPARAMS


class TestMockModel(test_utils.ModelTester):
    def get_model(self, mode):
        return MockModel(mode=mode, hparams=HPARAMS)


if __name__ == "__main__":
    tf.test.main()

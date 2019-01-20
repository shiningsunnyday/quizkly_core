""" Client for running sentence selector """
import tensorflow as tf

from models.sentence_classifier import SENTENCE_SIGNATURE, PREDICTION_SIGNATURE


class SentenceClassifierClient(object):
    """ Class to run sentence classifier model"""

    def __init__(self, model_dir):
        self._predictor = tf.contrib.predictor.from_saved_model(model_dir)

    def predict(self, sentences):
        """Serves predictions for given sentences
        Args:
            sentences: list of strings.
        """
        output_dict = self._predictor({SENTENCE_SIGNATURE: sentences})
        return output_dict[PREDICTION_SIGNATURE]

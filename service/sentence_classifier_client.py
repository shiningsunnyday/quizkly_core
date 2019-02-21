""" Client for running sentence selector """
import tensorflow as tf

from models.sentence_classifier import (
    SENTENCE_SIGNATURE, PREDICTION_SIGNATURE, CONTEXT_SIGNATURE)


class SentenceClassifierClient(object):
    """ Class to run sentence classifier model"""

    def __init__(self, model_dir):
        self._predictor = tf.contrib.predictor.from_saved_model(model_dir)

    def predict(self, sentences, contexts):
        """Serves predictions for given sentences
        Args:
            sentences: list of strings.
            contexts: list of strings indicating context of sentence.
        """
        output_dict = self._predictor(
            {SENTENCE_SIGNATURE: sentences, CONTEXT_SIGNATURE: contexts})
        return output_dict[PREDICTION_SIGNATURE]

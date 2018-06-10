""" Client for running sentence selector """
import tensorflow as tf

from models.sentence_classifier import SentenceClassifier
from models.sentence_classifier import SENTENCE_SIGNATURE, PREDICTION_SIGNATURE

class SentenceClassifierClient(object):
    """ Class to run sentence classifier model"""

    def __init__(self, model_dir):
        self._session = tf.Session(graph=tf.Graph())
        tf.saved_model.loader.load(
            self._session, ["inference"],
            model_dir)
        self._input_sentences = self._session.graph.get_tensor_by_name(
            SENTENCE_SIGNATURE + ":0")
        self._predictions =  self._session.graph.get_tensor_by_name(
            PREDICTION_SIGNATURE + ":0")

    def stop(self):
        """Frees up the resources used by the client."""
        self._session.close()

    def predict(self, sentences):
        """Serves predictions for given sentences
        Args:
            sentences: list of strings.
        """
        return self._session.run(
            self._predictions, {self._input_sentences: sentences})

""" Client for running distractor encoder """
import tensorflow as tf

from models.distractor_encoder import (
    QUESTION_SIGNATURE, ANSWER_SIGNATURE, ENCODING_SIGNATURE)


class DistractorEncoderClient(object):
    """ Class to run distractor encoder """

    def __init__(self, model_dir):
        self._predictor = tf.contrib.predictor.from_saved_model(model_dir)

    def predict(self, questions, answers):
        """Serves predictions for given sentences
        Args:
            sentences: list of strings.
        """
        output_dict = self._predictor(
            {QUESTION_SIGNATURE: questions, ANSWER_SIGNATURE: answers})
        return output_dict[ENCODING_SIGNATURE]

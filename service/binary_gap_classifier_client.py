"""Client for serving predictions by a keras binary gap classifier.
"""

import numpy as np
import tensorflow as tf


class BinaryGapClassifierClient(object):
    """Serves predictions by a keras binary encoding classifier.
    Args:
        model_dir: directory of saved model.
    """

    def __init__(self, model_path):
        """Create a new BinaryEncodingClassifierClient."""
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default(), self._session.as_default():
            self._model = tf.keras.models.load_model(model_path)

    def predict(self, encodings):
        """Returns a list of probabilities of the class to be predicted."""
        encodings = np.asarray(encodings, np.float32)
        with self._graph.as_default(), self._session.as_default():
            return self._model.predict(encodings)

    def choose_best_gaps(self, question_candidates):
        """Chooses the best gap for every question candidate."""
        gap_start_idx = [0]
        gap_embeddings = []
        for question_candidate in question_candidates:
            gap_embeddings.extend(
                [gap.embedding for gap in question_candidate.gap_candidates]
            )
            gap_start_idx.append(len(gap_embeddings))
        gap_predictions = self.predict(gap_embeddings)
        chosen_gaps = []
        for i, question_candidate in enumerate(question_candidates):
            best_gap_idx = np.argmax(
                gap_predictions[gap_start_idx[i]: gap_start_idx[i + 1]]
            )
            chosen_gap = question_candidate.gap_candidates[best_gap_idx]
            question_candidate.gap.MergeFrom(chosen_gap)
            chosen_gaps.append(chosen_gap)
        return chosen_gaps

    def stop(self):
        """Frees up the resources used by the client."""
        self._session.close()

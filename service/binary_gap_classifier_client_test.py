import tensorflow as tf

from proto.question_candidate_pb2 import QuestionCandidate
from service.binary_gap_classifier_client import BinaryGapClassifierClient


class BinaryGapClassifierClientTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        saved_model_path = "models/test_data/gap_classifier.h5"
        cls._client = BinaryGapClassifierClient(saved_model_path)
        cls._question_candidates = []
        for record in tf.python_io.tf_record_iterator(
            "datasets/testdata/question_candidates_test"
        ):
            question_candidate = QuestionCandidate()
            question_candidate.ParseFromString(record)
            cls._question_candidates.append(question_candidate)

    def test_choose_best_gaps(self):
        predictions = self._client.choose_best_gaps(self._question_candidates)
        for i, qc in enumerate(self._question_candidates):
            self.assertEqual(qc.gap.text, predictions[i].text)

    def test_predict(self):
        predictions = self._client.predict([[0.1, 0.2, 0.3], [0.23, 0.4, 5]])
        for (pred,) in predictions:
            self.assertLessEqual(pred, 1)
            self.assertLessEqual(0, pred)


if __name__ == "__main__":
    tf.test.main()

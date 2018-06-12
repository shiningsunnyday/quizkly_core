import tensorflow as tf

from service.sentence_classifier_client import SentenceClassifierClient


class SentenceClassifierClientTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._client = SentenceClassifierClient(
            "models/test_data/sentence_classifier/saved_model"
        )

    def test_predict(self):
        predictions = self._client.predict(["hey", "hi", "bye"])
        self.assertListEqual([3], list(predictions.shape))


if __name__ == "__main__":
    tf.test.main()

""" Tests for distractor encoder client. """
import tensorflow as tf

from service.distractor_encoder_client import DistractorEncoderClient


class DistractorEncoderClientTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._client = DistractorEncoderClient(
            "models/test_data/distractor_encoder/saved_model"
        )

    def test_predict(self):
        predictions = self._client.predict(
            ["hey", "hi", "bye"], ["bye", "hi", "hey"])
        self.assertListEqual([3, 10], list(predictions.shape))


if __name__ == "__main__":
    tf.test.main()

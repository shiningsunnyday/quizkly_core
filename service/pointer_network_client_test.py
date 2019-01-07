import tensorflow as tf

from service.pointer_network_client import PointerNetworkClient


class PointerNetworkClientTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._client = PointerNetworkClient(
            "models/test_data/pointer_network/saved_model"
        )

    def test_predict(self):
        predictions = self._client.predict(["hey", "hi", "bye"])
        self.assertListEqual([3], list(predictions[0].shape))
        self.assertListEqual([3], list(predictions[1].shape))


if __name__ == "__main__":
    tf.test.main()

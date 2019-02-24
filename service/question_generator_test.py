""" Tests for QuestionGenerator """
import spacy
import tensorflow as tf

from datasets.create_test_data import DummyElmoClient
from service.question_generator import QuestionGenerator


class QuestionGeneratorTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        gap_model_path = "models/test_data/gap_classifier.h5"
        sentence_model_path = ("models/test_data/sentence_classifier/"
                               "saved_model")
        word_model_path = "models/test_data/word_model"
        parser = spacy.load("en_core_web_sm")
        cls._client = QuestionGenerator(
            sentence_model_path, gap_model_path, word_model_path,
            DummyElmoClient(), parser)
        cls._text = "John went to the market. The market was huge."

    def test_generate_questions(self):
        batches = self._client.generate_questions(
            self._text, batch_size=2, is_test=True)
        for batch in batches:
            for i, qc in enumerate(batch):
                self.assertNotEqual(qc, None)


if __name__ == "__main__":
    tf.test.main()

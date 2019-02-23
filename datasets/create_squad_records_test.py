"""Test methods fopr creating squad tfrecords."""
import json
import re
import unittest

from datasets.create_squad_records import get_question_sentence_tuples


class TestCreateSquadRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_json_str = """
            {"qas": [
                {"question": "Who was Obama?",
                  "answers": [
                      {"text": "president", "answer_start": 10}
                   ]
                },
                {"question": "Where was Obama?",
                  "answers": [
                      {"text": "Washington DC", "answer_start": 37}
                   ]
                },
                {"question": "Who married Obama?",
                  "answers": [
                      {"text": "Michelle", "answer_start": 80}
                   ]
                }
             ],
             "context": "Obama was president. He was popular.
                         Washington DC was where he was.
                         He married Michelle in 1920.
                         He hated bad people."
            }
        """
        test_json_str = re.sub(r"\s\s+", " ", test_json_str)
        cls.squad_data = json.loads(test_json_str)

    def test_get_question_sentence_tuples(self):
        context = ("Obama was president. He was popular. "
                   "Washington DC was where he was. "
                   "He married Michelle in 1920. "
                   "He hated bad people.")
        expected_data_tuples = [
            (
                "Obama was president.",
                "Who was Obama?",
                "president",
                1,
                2,
                2,
                3,
                context
            ),
            ("He was popular.", "", "", 0, -1, -1, 3, context),
            (
                "Washington DC was where he was.",
                "Where was Obama?",
                "Washington DC",
                1,
                0,
                1,
                6,
                context
            ),
            (
                "He married Michelle in 1920.",
                "Who married Obama?",
                "Michelle",
                1,
                2,
                2,
                5,
                context
            ),
            ("He hated bad people.", "", "", 0, -1, -1, 4, context),
        ]
        self.assertListEqual(
            list(get_question_sentence_tuples(self.squad_data)),
            expected_data_tuples,
        )


if __name__ == "__main__":
    unittest.main()

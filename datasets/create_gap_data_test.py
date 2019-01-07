"""Test methods fopr creating QuesitonCandidate Protos."""

import unittest

from datasets.create_gap_data import create_question_candidates
from datasets.create_test_data import GAP_DATA_CONTAINER, DummyElmoClient
from proto.question_candidate_pb2 import Gap


class TestCreateQuestionCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._gap_text_lists = [
            ["White House", "Washington D.C."],
            ["New York", "Saturday", "Mr. Best"],
        ]

    def test_create_question_candidates(self):
        question_candidates = create_question_candidates(
            GAP_DATA_CONTAINER.docs, self._gap_text_lists, DummyElmoClient()
        )
        for i, qc in enumerate(question_candidates):
            self.assertListEqual(
                sorted(
                    [
                        g.text
                        for g in qc.gap_candidates
                        if g.train_label == Gap.POSITIVE
                    ]
                ),
                sorted(self._gap_text_lists[i]),
            )
            for g in qc.gap_candidates:
                self.assertEqual(3, len(g.embedding))


if __name__ == "__main__":
    unittest.main()

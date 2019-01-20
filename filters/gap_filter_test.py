"""Test methods for filtering gaps."""

import itertools
import unittest

from datasets.create_test_data import GAP_DATA_CONTAINER, DummyElmoClient
from filters.gap_filter import filter_gaps, filter_gaps_single_doc
from filters.gap_filter import backoff_phrase


class TestFilterGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._expected_cands = [
            ["White House", "Washington D.C.", "located"],
            ["Mr. Best", "New York", "Saturday", "morning", "flew"],
        ]

    def test_filter_gaps_single_doc(self):
        for i, doc in enumerate(GAP_DATA_CONTAINER.docs):
            gap_cands = filter_gaps_single_doc(doc).gap_candidates
            texts = [g.text for g in gap_cands]
            self.assertListEqual(texts, self._expected_cands[i])

    def test_filter_gaps(self):
        question_cands = list(
            *itertools.chain(
                filter_gaps(
                    GAP_DATA_CONTAINER.docs, elmo_client=DummyElmoClient()
                )
            )
        )
        for i, qc in enumerate(question_cands):
            texts = [g.text for g in qc.gap_candidates]
            self.assertListEqual(texts, self._expected_cands[i])
            for g in qc.gap_candidates:
                self.assertGreater(len(g.embedding), 0)

    def test_back_off_phrase(self):
        # TODO: WRITE A PROPER TEST @girish
        self.assertEqual(
            # to New York
            backoff_phrase(GAP_DATA_CONTAINER.docs[1][3:6]),
            GAP_DATA_CONTAINER.docs[1][4:6]
        )


if __name__ == "__main__":
    unittest.main()

"""Test methods for filtering gaps."""

import collections
import unittest
from unittest import mock

from nltk import stem
from nltk.corpus import wordnet as wn

from datasets.create_test_data import GAP_DATA_CONTAINER, DummyWordModel
from filters import distractor_filter
from proto import question_candidate_pb2


class TestFilterDistractors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sentences = [
            "The circulatory system pumps blood to all organs.",
            "The cytoplasm is the semiliquid portion of the cell.",
            "Proteins are manufactured by ribosomes",
            "Mr. Best flew to New York on Saturday morning.",
            "This process known as gene recombination helps in reproduction."
        ]
        gaps = [
            question_candidate_pb2.Gap(
                text="circulatory system", start_index=1, end_index=3,
                pos_tags=["ADJ", "NOUN"]
            ),
            question_candidate_pb2.Gap(
                text="cytoplasm", start_index=1, end_index=2,
                pos_tags=["NOUN"]
            ),
            question_candidate_pb2.Gap(
                text="Proteins", start_index=0, end_index=1,
                pos_tags=["NOUN"]
            ),
            question_candidate_pb2.Gap(
                text="Saturday", start_index=7, end_index=8,
                pos_tags=["NOUN"]
            ),
            question_candidate_pb2.Gap(
                text="recombination", start_index=5, end_index=6,
                pos_tags=["NOUN"]
            )
        ]
        cls._question_cands = [
            question_candidate_pb2.QuestionCandidate(
                question_sentence=sent,
                gap=gaps[i]
            )
            for i, sent in enumerate(sentences)
        ]
        cls._stemmer = stem.snowball.SnowballStemmer("english")
        cls._word_model = DummyWordModel()

    def test_nearest_neighbors(self):
        topn = 4
        candidates = distractor_filter.nearest_neighbors(
            self._question_cands[0].gap, self._word_model, topn)
        self.assertListEqual(
            candidates,
            [["a b", 1.0], ["b c", 0.9], ["c d", 0.8], ["d e", 0.7]])

    def test_filter_words_in_sent(self):
        distractors = [
            ("respiratory system", 0.90), ("blood", 0.90),
            ("organ", 0.8), ("digestive system", 0.7)
        ]
        filtered_distractors = distractor_filter.filter_words_in_sent(
            self._question_cands[0].gap,
            self._question_cands[0].question_sentence,
            distractors, self._stemmer)
        self.assertNotIn(("organ", 0.8), filtered_distractors)
        self.assertNotIn(("blood", 0.9), filtered_distractors)

        distractors = [
            ("respiratory system", 0.90), ("blood", 0.90),
            ("cells", 0.8), ("portions", 0.7)
        ]
        filtered_distractors = distractor_filter.filter_words_in_sent(
            self._question_cands[1].gap,
            self._question_cands[1].question_sentence,
            distractors, self._stemmer)
        self.assertNotIn(("cells", 0.8), filtered_distractors)
        self.assertNotIn(("portions", 0.7), filtered_distractors)

        distractors = [
            ("gene replication", 0.90), ("genetic recombination", 0.90),
            ("replication", 0.8), ("splitting", 0.7)
        ]
        filtered_distractors = distractor_filter.filter_words_in_sent(
            self._question_cands[4].gap,
            self._question_cands[4].question_sentence,
            distractors, self._stemmer)
        self.assertNotIn(("gene replication", 0.90), filtered_distractors)
        self.assertNotIn(("genetic recombination", 0.90), filtered_distractors)

    def test_filter_part_of_speech(self):

        class Syn(object):
            def __init__(self, pos):
                self._pos = pos

            def pos(self):
                return self._pos

        distractors = [
            ([Syn(wn.NOUN)], ("respiratory system", 0.90)),
            ([Syn(wn.ADJ)], ("cardiovascular", 0.90)),
            ([Syn(wn.NOUN)], ("organ", 0.8)),
            ([Syn(wn.ADV)], ("at", 0.7))
        ]
        for qc in self._question_cands:
            filtered_distractors = distractor_filter.filter_part_of_speech(
                qc.gap, distractors)
            self.assertNotIn(
                ([Syn(wn.ADJ)], ("cardiovascular", 0.90)),
                filtered_distractors)
            self.assertNotIn(
                ([Syn(wn.ADV)], ("at", 0.7)), filtered_distractors)

    def test_filter_wordnet(self):
        distractors = [
            ("respiratory system", 0.90), ("blood", 0.90),
            ("digestive system", 0.8), ("digestive system", 0.7),
            ("cardiovascular system", 0.8), ("blood vessels", 0.8),
            ("heart", 0.8), ("car", 0.7)
        ]
        filtered_distractors = distractor_filter.filter_wordnet(
            self._question_cands[0].gap, distractors, self._stemmer)
        self.assertListEqual(
            [("respiratory system", 0.90), ("digestive system", 0.8)],
            list(filtered_distractors))

    def test_filter_distractors(self):
        # Mr. Best flew to New York on Saturday morning.
        spacy_doc = GAP_DATA_CONTAINER.docs[1]
        qc = self._question_cands[3]
        mock_parser = mock.Mock()
        mock_parser.tokenizer = mock.Mock()
        mock_parser.tokenizer.pipe = mock.MagicMock(return_value=None)
        mock_parser.tagger = mock.Mock()
        TaggedTuple = collections.namedtuple("TaggedTuple", ["pos_"])
        mock_parser.tagger.pipe = mock.MagicMock(
            return_value=iter(
                [
                    [TaggedTuple("NOUN")]
                    for _ in range(20)]
            )
        )
        distractor_filter.filter_distractors_single(
            qc, spacy_doc, mock_parser, self._word_model, self._stemmer)


if __name__ == "__main__":
    unittest.main()

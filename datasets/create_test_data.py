""" Script to create test data for tests. """

import numpy
from spacy.util import get_lang_class
from spacy.tests.util import get_doc
import tensorflow as tf

from datasets import create_squad_records
from datasets.create_gap_data import create_question_candidates
from datasets.dataset_utils import create_example


class GapDataContainer(object):
    """ Dummy class to contain data for gap-data related tests"""

    def __init__(self):
        sentences = [
            "the White House is located in Washington D.C.",
            "Mr. Best flew to New York on Saturday morning.",
        ]
        en_tokenizer = get_lang_class("en").Defaults.create_tokenizer()
        self.docs = []
        tokens = [en_tokenizer(s) for s in sentences]
        vocabs = [tok.vocab for tok in tokens]
        tags = [
            ["DT", "NNP", "NNP", "VBZ", "VBN", "IN", "NNP", "NNP"],
            ["NNP", "NNP", "VBD", "IN", "NNP", "NNP", "IN", "NNP", "NN", "."],
        ]
        pos = [
            ["DET", "PROPN", "PROPN", "VERB", "VERB", "ADP", "PROPN", "PROPN"],
            [
                "PROPN",
                "PROPN",
                "VERB",
                "ADP",
                "PROPN",
                "PROPN",
                "ADP",
                "PROPN",
                "NOUN",
                "PUNCT",
            ],
        ]
        heads = [
            [2, 1, 2, 1, 0, -1, -1, -1],
            [1, 1, 0, -1, 1, -2, -4, 1, -2, -7],
        ]
        deps = [
            [
                "det",
                "compound",
                "nsubjpass",
                "auxpass",
                "ROOT",
                "prep",
                "pobj",
                "appos",
            ],
            [
                "compound",
                "nsubj",
                "ROOT",
                "prep",
                "compound",
                "pobj",
                "prep",
                "compound",
                "pobj",
                "punct",
            ],
        ]
        ents = [
            [
                ("the White House", "ORG", 0, 3),
                ("Washington D.C.", "GPE", 6, 8),
            ],
            [
                ("Mr. Best", "PERSON", 0, 2),
                ("New York", "GPE", 4, 6),
                ("Saturday", "DATE", 7, 8),
                ("morning", "TIME", 8, 9),
            ],
        ]
        self._expected_cands = [
            ["White House", "Washington D.C.", "located"],
            ["Mr. Best", "New York", "Saturday", "morning", "flew"],
        ]
        for i in range(len(sentences)):
            self.docs.append(
                get_doc(
                    vocabs[i],
                    [t.text for t in tokens[i]],
                    pos=pos[i],
                    heads=heads[i],
                    deps=deps[i],
                    tags=tags[i],
                    ents=ents[i],
                )
            )


GAP_DATA_CONTAINER = GapDataContainer()


class DummyElmoClient(object):
    """A dummy client that encodes text as random vectors."""

    def encode(self, texts):
        max_len = 0
        for text in texts:
            split_text = text.split(" ")
            if len(split_text) > max_len:
                max_len = len(split_text)
        return numpy.random.normal(size=(len(texts), max_len, 3))


if __name__ == "__main__":
    # Create test squad data records.
    squad_tuples = [
        ("Obama was president.", "Who was Obama?", "president", 1, 2, 3, 3),
        ("He was popular.", "", "", 0, -1, -1, 4),
        (
            "Washington DC was where he was.",
            "Where was Obama?",
            "Washington DC",
            1,
            0,
            2,
            6,
        ),
        (
            "He married Michelle in 1920.",
            "Who married Obama?",
            "Michelle",
            1,
            2,
            3,
            5,
        ),
        ("He hated bad people.", "", "", 0, -1, -1, 4),
    ]
    tfrecord_writer = tf.python_io.TFRecordWriter(
        "datasets/testdata/squad_test.tfrecords"
    )
    for tup in squad_tuples:
        tf_example = create_example(tup, create_squad_records.FEATURE_NAMES)
        tfrecord_writer.write(tf_example.SerializeToString())
    tfrecord_writer.close()

    # Create test question candidates.
    gap_text_lists = [
        ["White House", "Washington D.C."],
        ["New York", "Saturday", "Mr. Best"],
    ]
    question_candidates = create_question_candidates(
        GAP_DATA_CONTAINER.docs, gap_text_lists, DummyElmoClient()
    )
    tfrecord_writer = tf.python_io.TFRecordWriter(
        "datasets/testdata/question_candidates_test"
    )
    for qc in question_candidates:
        tfrecord_writer.write(qc.SerializeToString())
    tfrecord_writer.close()

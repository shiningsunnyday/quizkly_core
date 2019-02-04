"""Test methods for filtering context sentences."""

import unittest

import spacy

from filters.context_filter import dep_context


class TestFilterGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parser = spacy.load("en_core_web_md", add_vectors=False)
        sentence_sets = [
            ["The tricuspid valve has three cusps, which connect to chordae "
             "tendinae and three papillary muscles named the anterior, "
             "posterior, and septal muscles, after their relative positions.",
             "The mitral valve lies between the left atrium and left "
             "ventricle.",
             "It is also known as the bicuspid valve due to its "
             "having two cusps, an anterior and a posterior cusp."],
            ["John went to the market.", "He bought snacks.",
             "He loves chocolates."],
            ["John is an animal lover"],
            ["He is a great person", "He is him", "He is very interesting."],
        ]
        cls._reference_sent_idxs = [2, ]
        cls._doc_sets = [list(parser.pipe(sents, n_threads=4))
                         for sents in sentence_sets]
        cls._expected = [
            "The mitral valve lies between the left atrium and left "
            "ventricle.",
            "John went to the market. He bought snacks.",
            "",
            ""
        ]

    def test_dep_context(self):
        for i, docs in enumerate(self._doc_sets):
            self.assertEqual(
                self._expected[i], dep_context(docs[:-1], docs[-1]))


if __name__ == "__main__":
    unittest.main()

"""Test methods for preprocessing text."""

import unittest

from filters.preprocessor import preprocess_text


class TestPreprocessor(unittest.TestCase):

    def test_preprocess_text(self):
        test_text = (
            "Barack Hussein Obama II (/bəˈrɑːk huːˈseɪn oʊˈbɑːmə/"
            " (About this soundlisten);[1]"
            " born August 4, 1961) is an American attorney and "
            "politician who served as the 44th president of the United"
            " States from 2009 to 2017[1]. A member of the Democratic"
            " Party, he was the first African American to be elected"
            " to the presidency[2]. He previously served as a U.S."
            " senator from Illinois from 2005 to 2008.")
        expected_text = (
            "Barack Hussein Obama II is an American attorney and "
            "politician who served as the 44th president of the United"
            " States from 2009 to 2017. A member of the Democratic "
            "Party, he was the first African American to be elected"
            " to the presidency. He previously served as a U.S."
            " senator from Illinois from 2005 to 2008.")
        self.assertEqual(preprocess_text(test_text), expected_text)


if __name__ == "__main__":
    unittest.main()

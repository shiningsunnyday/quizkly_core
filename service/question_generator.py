"""Script to extract question worthy sentences from a file."""
import argparse

import gensim
import nltk
import spacy

from filters import distractor_filter, gap_filter
from service.binary_gap_classifier_client import (
    BinaryGapClassifierClient)
from service.elmo_client import ElmoClient
from service.sentence_classifier_client import SentenceClassifierClient

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument(
    "--input_file",
    required=True,
    help="Location of file with text.",
)
_PARSER.add_argument(
    "--sentence_model",
    required=True,
    help="Directory of saved sentence model to use.",
)
_PARSER.add_argument(
    "--gap_model",
    required=True,
    help="Directory of saved gap model to use.",
)
_PARSER.add_argument(
    "--word_model",
    required=True,
    help="Directory of saved word model to use.",
)
_PARSER.add_argument(
    "--output_file",
    help="Directory to save sentences with questions.",
)
_PARSER.add_argument(
    "--batch_size",
    default=100,
    help="Size of batches to compute predictions for.",
)
_FLAGS = _PARSER.parse_args()

class QuestionGenerator(object):
    def __init__(self, sentence_model_path, gap_model_path, word_model_path):
        self._sentence_client = SentenceClassifierClient(_FLAGS.sentence_model)
        self._gap_client = BinaryGapClassifierClient(_FLAGS.gap_model)
        self._elmo_client = ElmoClient()
        self._word_model = (
            gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
                _FLAGS.word_model, binary=True)
        )
        self._parser = spacy.load("en_core_web_md")
        # Handle spacy bug with stopwords.
        self._parser.vocab.add_flag(
            lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
            spacy.attrs.IS_STOP)

    def generate_questions(text, batch_size=50):
        sentences = nltk.sent_tokenize(text)
        predictions = sentence_client.predict(batch)
        i = 0
        while i < len(sentences):
            chosen_sents = []
            batch = sentences[i: i + _FLAGS.batch_size]
            predictions = self._sentence_client.predict(batch)
            for p in predictions:
                if p > 0:
                    chosen_sents.append(p)
            elmo_client = ElmoClient()
            spacy_docs = list(parser.pipe(batch, n_threads=4))
            question_candidates, = list(gap_filter.filter_gaps(
                spacy_docs, batch_size=len(spacy_docs),
                elmo_client=self._elmo_client))
            gap_client.choose_best_gaps(question_candidates)
            distractor_filter.filter_distractors(
                question_candidates, spacy_docs, parser, word_model)
            i += _FLAGS.batch_size
            yield question_candidates


def _main():
    question_generator = QuestionGenerator(
        _FLAGS.sentence_model, _FLAGS.gap_model, _FLAGS.word_model)
    with open(_FLAGS.input_file) as f:
        text = f.read()
    question_candidates = []
    for batch in question_generator.generate_questions(text):
        question_candidates.extend(batch)

    print("Writing to output file...")
    output_file = _FLAGS.output_file or _FLAGS.input_file
    with open(output_file, "w") as f:
        for qc in question_candidates:
            answer = qc.gap.text
            question = qc.question_sentence.replace(
                answer, "_________"
            )
            distractors = "\n".join(
                "%d. %s" % (i, dist.text)
                for i, dist in enumerate(qc.distractors))
            f.write(question)
            f.write("\n")
            f.write(answer)
            f.write("\n")
            f.write(distractors)
            f.write("\n*********\n")
    print("Done!")


if __name__ == "__main__":
    _main()

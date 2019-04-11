"""Script to run full question generation pipeline on a file.
Usage:
    python -m service.question_generator \
        --input_file test.txt
        --sentence_model models/saved_sentence_model
        --gap_model models/gap_model
        --word_model models/model.word2vec
        --output_file test_qns.txt
        --batch_size 100
"""
import argparse

import gensim
import nltk
import spacy

from filters import context_filter, distractor_filter, gap_filter, preprocessor
from service.binary_gap_classifier_client import (
    BinaryGapClassifierClient)
from service.elmo_client import ElmoClient
from service.sentence_classifier_client import SentenceClassifierClient


class QuestionGenerator(object):
    def __init__(self, sentence_model_path, gap_model_path,
                 word_model_path, elmo_client, parser):
        self._sentence_client = SentenceClassifierClient(sentence_model_path)
        self._gap_client = BinaryGapClassifierClient(gap_model_path)
        self._elmo_client = elmo_client
        self._word_model = (
            gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
                word_model_path, binary=True)
        )
        self.parser = parser

        # Handle spacy bug with stopwords.
        self.parser.vocab.add_flag(
            lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
            spacy.attrs.IS_STOP)

    def generate_questions(self, text, batch_size=50, context_window=5,
                           is_test=False):
        text = preprocessor.preprocess_text(text)
        sentences = nltk.sent_tokenize(text)
        i = 0
        while i < len(sentences):
            chosen_sent_idxs = []
            batch = sentences[i: i + batch_size]
            contexts = [
                " ".join(sentences[max(j - context_window, 0):
                                   min(j + context_window, len(sentences))])
                for j in range(i, i + len(batch))
            ]
            predictions = self._sentence_client.predict(
                batch, contexts)
            for j, p in enumerate(predictions):
                if p > 0 or is_test:
                    chosen_sent_idxs.append(j)
            if len(chosen_sent_idxs) == 0:
                continue
            spacy_docs = list(self.parser.pipe(batch, n_threads=4))
            chosen_docs = [spacy_docs[j] for j in chosen_sent_idxs]
            question_candidates, = list(gap_filter.filter_gaps(
                chosen_docs, batch_size=len(chosen_docs),
                elmo_client=self._elmo_client))
            self._gap_client.choose_best_gaps(question_candidates)
            distractor_filter.filter_distractors(
                question_candidates, chosen_docs,
                self.parser, self._word_model)
            for j, qc in enumerate(question_candidates):
                context_filter.dep_context(
                    spacy_docs[:chosen_sent_idxs[j]],
                    spacy_docs[j],
                    self.parser)
            i += batch_size
            yield question_candidates


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        help="Location of file with text.",
    )
    parser.add_argument(
        "--sentence_model",
        required=True,
        help="Directory of saved sentence model to use.",
    )
    parser.add_argument(
        "--gap_model",
        required=True,
        help="Directory of saved gap model to use.",
    )
    parser.add_argument(
        "--word_model",
        required=True,
        help="Directory of saved word model to use.",
    )
    parser.add_argument(
        "--output_file",
        help="Directory to save sentences with questions.",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        help="Size of batches to compute predictions for.",
    )
    flags = parser.parse_args()
    elmo_client = ElmoClient()
    question_generator = QuestionGenerator(
        flags.sentence_model, flags.gap_model,
        flags.word_model, elmo_client,
        spacy.load("en_core_web_md"))
    with open(flags.input_file) as f:
        text = f.read()
    question_candidates = []
    for batch in question_generator.generate_questions(text):
        question_candidates.extend(batch)

    print("Writing to output file...")
    output_file = flags.output_file or flags.input_file
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

"""Script to extract question worthy sentences from a file."""
import argparse

import spacy

from filters import gap_filter
from service.elmo_client import ElmoClient
from service.binary_gap_classifier_client import (
    BinaryGapClassifierClient)

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument(
    "--input_file",
    required=True,
    help="Location of file with line-separated input sentences.",
)
_PARSER.add_argument(
    "--saved_model",
    required=True,
    help="Directory of saved model to use.",
)
_PARSER.add_argument(
    "--output_file",
    help="Directory to save sentences with chosen gaps.",
)
_PARSER.add_argument(
    "--batch_size",
    default=100,
    help="Size of batches to compute predictions for.",
)
_FLAGS = _PARSER.parse_args()


def _get_question_candidates(lines):
    elmo_client = ElmoClient()
    nlp = spacy.load("en_core_web_md")
    nlp.vocab.add_flag(
        lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
        spacy.attrs.IS_STOP)
    spacy_docs = list(
        nlp.pipe(lines, n_threads=4))
    question_candidates = []
    batches = gap_filter.filter_gaps(
        spacy_docs, batch_size=50, elmo_client=elmo_client)
    for batch in batches:
        question_candidates.extend(batch)
    return question_candidates


def _main():
    print("Loading Model...")
    client = BinaryGapClassifierClient(
        _FLAGS.saved_model)
    with open(_FLAGS.input_file) as f:
        lines = f.readlines()
    question_candidates = _get_question_candidates(lines)
    print("Getting Predictions...")
    i = 0
    while i < len(question_candidates):
        client.choose_best_gaps(question_candidates[i: i + _FLAGS.batch_size])
        i += _FLAGS.batch_size
    print("Writing to output file...")
    output_file = _FLAGS.output_file or _FLAGS.input_file
    with open(output_file, "w") as f:
        for qc in question_candidates:
            answer = qc.gap.text
            question = qc.question_sentence.replace(
                answer, "_________"
            )
            f.write(question)
            f.write("\n")
            f.write(answer)
            f.write("*********\n")
    print("Done!")


if __name__ == "__main__":
    _main()

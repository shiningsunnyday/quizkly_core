"""Script to extract question worthy sentences from a file.

Usage:
    python -m service.question_sentence_selector \
        --input_file test.txt
        --saved_model models/saved_sentence_model
        --output_file test_qns.txt
        --batch_size 100
"""

import argparse

from service.sentence_classifier_client import SentenceClassifierClient

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument(
    "--input_file",
    required=True,
    help="Location of file with line-separated input sentences.",
)
_PARSER.add_argument(
    "--saved_model",
    required=True,
    help="Directory of saved model to use to compute the encodings.",
)
_PARSER.add_argument(
    "--output_file",
    help="Directory to save the index. If None, overwrites input file.",
)
_PARSER.add_argument(
    "--batch_size",
    default=100,
    help="Size of batches to compute predictions for.",
)
_FLAGS = _PARSER.parse_args()


def _main():
    print("Loading Model...")
    client = SentenceClassifierClient(_FLAGS.saved_model)
    with open(_FLAGS.input_file) as f:
        lines = f.readlines()
    print("Getting Predictions...")
    predictions = []
    i = 0
    while i < len(lines):
        predictions.extend(
            client.predict(lines[i: i + _FLAGS.batch_size]))
        i += _FLAGS.batch_size
    print("Writing to output file...")
    output_file = _FLAGS.output_file or _FLAGS.input_file
    with open(output_file, "w") as f:
        for i, line in enumerate(lines):
            if predictions[i] > 0:
                f.write(line)
    print("Done!")


if __name__ == "__main__":
    _main()

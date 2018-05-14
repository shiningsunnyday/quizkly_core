"""
Script to generate tfrecords from SQuAD json files.
"""
from argparse import ArgumentParser
import json
import nltk
import re
import string
import tensorflow as tf

from dataset_utils import create_example

SENTENCE = "sentence"
QUESTION = "question"
ANSWER = "answer"
QUESTION_WORTHY_LABEL = "question_worthy"
START_INDEX = "answer_start"
END_INDEX = "answer_end"

FEATURE_NAMES = [
    SENTENCE, QUESTION, ANSWER, QUESTION_WORTHY_LABEL,
    START_INDEX, END_INDEX
]


def get_question_sentence_tuples(paragraph_data):
    """ Returns a question, sentence, answer, question-worthy label, tuple.
    Args:
        paragraph_data: Each "paragraph" as in the squad json file.

    Return:
        a list of tuples containing the initial sentence, crafted question
        (if it exists), answer to the question if question exists,
        question-worthy label (1 if question exists, 0 otherwise),
        start index of answer tokens (inclusive) and 
        end index of answer tokens (exclusive).
    """
    processed_tuples = []
    context = paragraph_data["context"]
    raw_sentences = nltk.sent_tokenize(context)
    sentence_idx = [context.find(sent) for sent in raw_sentences]
    for i, sent in enumerate(raw_sentences):
        split_sent = [
            re.sub('['+string.punctuation+']', '', s)
            for s in sent.split()
        ]
        question_worthy = False
        for qa in paragraph_data["qas"]:
            question = qa["question"]
            answers = qa["answers"]
            for answer in answers:
                answer_idx = answer["answer_start"]
                ans_text = answer["text"]
                if (sentence_idx[i] <= answer_idx and
                        sentence_idx[i] + len(sent) > answer_idx):
                    question_worthy = True
                    start_idx = split_sent.index(ans_text)
                    end_idx = start_idx + len(ans_text.split())
                    yield (sent, question, ans_text, 1, start_idx, end_idx)
        if not question_worthy:
            yield (sent, "", "", 0, -1, -1)

def _write_tf_records(args):
    with open(args.input_file) as f:
        json_data = json.loads(f.read())
    options = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    tfrecord_writer = tf.python_io.TFRecordWriter(
        filename, options=options)

    for wiki_page in json_data['data']:
        for para_data in wiki_page['paragraphs']:
            data_tuples = get_question_sentence_tuples(para_data)
            for tup in data_tuples:
                tf_example = create_example(tup, FEATURE_NAMES)
                tfrecord_writer.write(tf_example.SerializeToString())
    tfrecord_writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        help="Location of json file containing squad data.")

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to output tfrecords.")

    args = parser.parse_args()
    _write_tf_records(args)

    




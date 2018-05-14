"""
Script to generate tfrecords from SQuAD json files.
"""
from argparse import ArgumentParser
import json
import nltk
import tensorflow as tf

from dataset_utils import create_example, strip_punctuation

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

    def _find_sub_list(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e == sl[0]):
            if l[ind: ind + sll] == sl:
                return (ind, ind+sll)
        return None

    context = paragraph_data["context"]
    raw_sentences = nltk.sent_tokenize(context)
    sentence_idx = [context.find(sent) for sent in raw_sentences]
    for i, sent in enumerate(raw_sentences):
        split_sent = [strip_punctuation(s) for s in sent.split()]
        question_worthy = False
        for qa in paragraph_data["qas"]:
            question = qa["question"]
            answers = qa["answers"]
            for answer in answers:
                answer_idx = answer["answer_start"]
                ans_text = answer["text"]
                split_ans = [strip_punctuation(a) for a in ans_text.split()]
                if (sentence_idx[i] <= answer_idx and
                        sentence_idx[i] + len(sent) > answer_idx):
                    question_worthy = True
                    ans_range = _find_sub_list(split_ans, split_sent)
                    if ans_range:
                        start_idx, end_idx = ans_range
                        yield (sent, question, ans_text, 1, start_idx, end_idx)
                    else:
                        yield (sent, question, ans_text, 1, -1, -1)
        if not question_worthy:
            yield (sent, "", "", 0, -1, -1)


def _write_tf_records(args):
    with open(args.input_file) as f:
        json_data = json.loads(f.read())
    options = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.ZLIB)
    tfrecord_writer = tf.python_io.TFRecordWriter(
        args.output_file, options=options)
    total_examples = 0
    num_qn_worthy = 0  # Number of question-worthy examples.
    # Number of examples where an exact match couldn't find an answer.
    # Needs a fix soon.
    num_no_ans = 0
    for wiki_page in json_data['data']:
        for para_data in wiki_page['paragraphs']:
            data_tuples = get_question_sentence_tuples(para_data)
            for tup in data_tuples:
                if tup[3] == 1:
                    num_qn_worthy += 1
                if tup[1] != "" and tup[5] == -1:
                    num_no_ans += 1
                tf_example = create_example(tup, FEATURE_NAMES)
                tfrecord_writer.write(tf_example.SerializeToString())
                total_examples += 1
    tf.logging.info("Total Number of Examples: %d" % total_examples)
    tf.logging.info(
        "Number of Question-Worthy Sentence Examples: %d" % num_qn_worthy)
    tf.logging.info(
        "Number of Question-Unworthy Sentence Examples: %d" %
        (total_examples - num_qn_worthy)
    )
    tf.logging.info(
        "Examples where an exact string match couldn't find an answer: %d" %
        (num_no_ans))
    tfrecord_writer.close()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file",
        required=True,
        help="Location of json file containing squad data.")

    parser.add_argument(
        "--output_file",
        required=True,
        help="File to output tfrecords.")

    args = parser.parse_args()
    _write_tf_records(args)

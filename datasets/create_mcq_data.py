""" Script to create tf sequence-examples for distractor ranking model. """
from argparse import ArgumentParser
import gensim
import os
import random
import tensorflow as tf

from datasets.dataset_utils import (
    bytes_feature, float_list_feature, int64_feature, write_to_file)


def _get_question_tuples(filepath):
    with open(filepath, "r") as f:
        contents = f.read().strip()
    for question_set in contents.split("#Q"):
        components = question_set.split("^ ")
        if len(components) != 2:
            continue
        question, options = components
        question = question.strip()
        options = options.strip().split("\n")
        answer = options[0].strip()
        distractors = []
        for option in options[1:]:
            opt = option.strip()[2:]
            if opt != answer:
                distractors.append(opt)
        yield (question, answer, distractors)


def create_sequence_examples(question, answer, distractors, word_model):
    """ Creates multiple sequence examples for distractor encoder model
    Args:
    question: string containing question
    answer: string containing answer
    distractors: list of strings containing distractors

    Yields:
    tf.train.SequenceExample
    """
    try:
        sampled_negatives = word_model.most_similar_cosmul(
            positive=[answer], topn=200)
    except KeyError:
        return
    distractors_lower = set([d.lower() for d in distractors])
    sampled_negatives = [
        word_model[w] for w, _ in sampled_negatives
        if w.lower() not in distractors_lower
    ][:100]

    for d in distractors:
        try:
            dist_vector = word_model[d]
        except KeyError:
            continue
        question_feature = bytes_feature(question.encode("utf-8"))
        answer_feature = bytes_feature(answer.encode("utf-8"))
        distractor_feature = float_list_feature(list(dist_vector))
        negative_features = [
            float_list_feature(list(n)) for n in sampled_negatives
        ]
        context = tf.train.Features(
            feature={
                "question": question_feature, "answer": answer_feature,
                "distractor": distractor_feature,
                "negatives_length": int64_feature(len(negative_features)),
            }
        )
        feature_list = {
            "negatives": tf.train.FeatureList(feature=negative_features)
        }
        yield tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list=feature_list),
            context=context)


def _write_tf_records(args):
    sequence_examples = []
    word_model = (
        gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
            args.word_model, binary=True)
    )
    for f in tf.gfile.Glob(args.input_file_pattern):
        for tup in _get_question_tuples(f):
            sequence_examples.extend(
                list(create_sequence_examples(*tup, word_model)))
    random.shuffle(sequence_examples)
    num_train = int(args.train_split * len(sequence_examples))
    train_examples = sequence_examples[:num_train]
    test_examples = sequence_examples[num_train:]
    tf.logging.info("Total Number of Train Examples: %d" % len(train_examples))
    tf.logging.info("Total Number of Test Examples: %d" % len(test_examples))
    write_to_file(
        train_examples, os.path.join(args.output_path, "train.tfrecords")
    )
    write_to_file(
        test_examples, os.path.join(args.output_path, "test.tfrecords")
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file_pattern",
        required=True,
        help="Location of files containing MCQ data.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help=(
            "Path of file to output tfrecords to. path/test.tfrecords"
            " and path/train.tfrecords"
        ),
    )
    parser.add_argument(
        "--train_split",
        default=0.8,
        help=(
            "Percentage of records to go into the train file."
            "Remainder goes to test file."
        ),
        type=float
    )
    parser.add_argument(
        "--word_model",
        required=True,
        help="Directory of saved word model to use.",
    )
    args = parser.parse_args()
    _write_tf_records(args)

""" Script to generate gap training data (QuestionCandidate protos). """
import argparse
import glob
import itertools
import logging
import os
import pickle
import spacy
import tensorflow as tf

from datasets.dataset_utils import write_to_file
from filters import gap_filter
from proto.question_candidate_pb2 import Gap
from service.elmo_client import ElmoClient


def create_question_candidates(
    docs, gap_text_lists, elmo_client=None, max_gap_length=3
):
    question_candidates = list(
        itertools.chain.from_iterable(
            gap_filter.filter_gaps(
                docs, batch_size=50, elmo_client=elmo_client
            )
        )
    )
    total_gaps = 0
    total_question_cands = 0
    positive_gaps = 0
    negative_gaps = 0
    for gap_text_list, question_candidate in zip(
        gap_text_lists, question_candidates
    ):
        total_gaps += len(question_candidate.gap_candidates)
        positive_gap_indices = set([])
        for gap_text in gap_text_list:
            if len(gap_text.split(" ")) > max_gap_length:
                continue
            for i, gap in enumerate(question_candidate.gap_candidates):
                if gap_text in gap.text:
                    positive_gap_indices.add(i)
        for i, gap in enumerate(question_candidate.gap_candidates):
            if i in positive_gap_indices:
                question_candidate.gap_candidates[i].train_label = Gap.POSITIVE
                positive_gaps += 1
            else:
                question_candidate.gap_candidates[i].train_label = Gap.NEGATIVE
                negative_gaps += 1
        total_question_cands += 1
    logging.info("Number of Question Candidates: %d" % total_question_cands)
    logging.info("Number of Gap Candidates: %d" % total_gaps)
    logging.info("Number of Positive Gap Candidates: %d" % positive_gaps)
    logging.info("Number of Negative Gap Candidates: %d" % negative_gaps)
    return question_candidates


def _get_spacy_docs_gap_text_lists(gap_tuples, nlp, max_doc_length=40):
    gap_tuples = [
        gap_tuple
        for gap_tuple in gap_tuples
        if len(gap_tuple[0].split(" ")) < max_doc_length
    ]
    sentences, _, gap_text_lists = zip(*gap_tuples)
    return list(nlp.pipe(sentences, n_threads=4)), gap_text_lists


def _get_squad_data(tfrecords_paths):
    squad_data = []
    example = tf.train.Example()
    for record_path in tfrecords_paths:
        for record in tf.python_io.tf_record_iterator(record_path):
            example.ParseFromString(record)
            feature_dict = example.features.feature
            sentence = (
                feature_dict["sentence"].bytes_list.value[0].decode("utf-8")
            )
            answer = feature_dict["answer"].bytes_list.value[0].decode("utf-8")
            if len(sentence) > 1 and len(answer) > 1:
                squad_data.append([sentence, "", [answer]])
    return squad_data


def _main(args):
    elmo_client = ElmoClient()
    nlp = spacy.load("en_core_web_md")
    nlp.vocab.add_flag(
        lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
        spacy.attrs.IS_STOP)
    if args.quizlet_pickle_patterns:
        quizlet_data = []
        quizlet_files = glob.glob(args.quizlet_pickle_patterns)
        for fname in quizlet_files:
            with open(fname, "rb") as f:
                quizlet_data += pickle.load(f)
        quizlet_candidates = create_question_candidates(
            *_get_spacy_docs_gap_text_lists(quizlet_data, nlp), elmo_client
        )
        write_to_file(
            quizlet_candidates,
            os.path.join(args.output_path, "quizlet.candidates"),
        )

    if args.squad_tfrecords:
        squad_tuples = _get_squad_data(glob.glob(args.squad_tfrecords))
        squad_candidates = create_question_candidates(
            *_get_spacy_docs_gap_text_lists(squad_tuples, nlp), elmo_client
        )
        write_to_file(
            squad_candidates,
            os.path.join(args.output_path, "squad.candidates"),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quizlet_pickle_patterns",
        default=None,
        help="File pattern of quizlet pickle files.",
    )
    parser.add_argument(
        "--squad_tfrecords",
        default=None,
        help="File pattern of squad tfrecords.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help=("Path of folder to output QuestionCandidate protos to."),
    )
    args = parser.parse_args()
    _main(args)

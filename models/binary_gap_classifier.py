"""Functions for training a binary gap classifier."""
import argparse
import hashlib
import logging
import random

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf

from proto.question_candidate_pb2 import QuestionCandidate
from proto.question_candidate_pb2 import Gap


_DEFAULT_HPARAMS = tf.contrib.training.HParams(
    learning_rate=0.9, hidden_size=100, encoding_dim=1024
)


def get_compiled_keras_model(hparams):
    """Create a compiled keras model for classifying encodings.
    Args:
        hparams: HParams with encoding_dim, hidden_size and
            learning_rate specified.
    Returns:
        a compiled keras model.
    """
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dropout(0.5, input_shape=(hparams.encoding_dim,))
    )
    model.add(tf.keras.layers.Dense(hparams.hidden_size, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(lr=hparams.learning_rate),
        metrics=["accuracy"],
    )
    return model


def train_model(hparams, records_pattern, output_path):
    """ Trains and saves a binary classifier on embeddings.
    Args:
        hparams: HParams with encoding_dim, hidden_size and
            learning_rate specified.
        pattern: pattern of QuestionCandidate protos.
        output_path: path to save model to.
    """
    positives, negatives = _get_data(records_pattern)
    logging.info("%d Positives" % len(positives))
    logging.info("%d negatives" % len(negatives))
    split_data = _split_train_test(positives, negatives)
    train_set, train_labels, test_set, test_labels = split_data
    logging.info("%d Training Samples" % len(train_set))
    logging.info("%d Testing Samples" % len(test_set))
    keras_model = get_compiled_keras_model(hparams)
    logging.info("Training model...")
    keras_model.fit(
        np.array(train_set),
        train_labels,
        batch_size=100,
        epochs=100,
        verbose=0,
    )
    logging.info("Evaluating...")
    evaluate_model(keras_model, np.array(test_set), test_labels)
    keras_model.save(output_path)
    logging.info("Saved model at %s" % output_path)


def evaluate_model(model, test_encodings, test_labels):
    """ Evaluates a trained keras binary classifier.
    Args:
        model: a trained keras binary classifier.
        test_encodings: numpy array of encodings.
        test_labels: numpy array of labels.
    """
    test_preds = model.predict(test_encodings)
    accuracy = accuracy_score(test_labels, test_preds >= 0.5)
    auc_score = roc_auc_score(test_labels, test_preds)
    logging.info("Accuracy: %.2f" % accuracy)
    logging.info("AUC: %.2f" % auc_score)


def _get_data(records_pattern):
    positives, negatives = [], []
    total_records = 0
    for record_path in tf.gfile.Glob(records_pattern):
        for record in tf.python_io.tf_record_iterator(record_path):
            question_candidate = QuestionCandidate()
            question_candidate.ParseFromString(record)
            pos, neg = [], []
            for gap in question_candidate.gap_candidates:
                if gap.train_label == Gap.POSITIVE:
                    pos.append(
                        (
                            question_candidate.question_sentence,
                            list(gap.embedding),
                        )
                    )
                elif gap.train_label == Gap.NEGATIVE:
                    neg.append(
                        (
                            question_candidate.question_sentence,
                            list(gap.embedding),
                        )
                    )
            if len(neg) > len(pos):
                positives.extend(pos)
                negatives.extend(random.sample(neg, len(pos)))
            else:
                positives.extend(neg)
                negatives.extend(random.sample(pos, len(neg)))
            total_records += 1
    logging.info("Processed %d QuestionCandidates" % total_records)
    return positives, negatives


def _split_train_test(positives, negatives, train_percent=80):
    def _hash_percent(string):
        return int(hashlib.md5(string.encode("utf-8")).hexdigest(), 16) % 100

    # Guarantee that both the train and test set have at least
    # one positive and one negative.
    train_set = [(positives[0][1], 1), (negatives[0][1], 0)]
    test_set = [(positives[1][1], 1), (negatives[1][1], 0)]
    for gap_text, embedding in positives:
        if _hash_percent(gap_text) < train_percent:
            train_set.append((embedding, 1))
        else:
            test_set.append((embedding, 1))
    for gap_text, embedding in negatives:
        if _hash_percent(gap_text) < train_percent:
            train_set.append((embedding, 0))
        else:
            test_set.append((embedding, 0))
    train_examples, train_labels = list(map(list, zip(*train_set)))
    test_examples, test_labels = list(map(list, zip(*test_set)))
    return train_examples, train_labels, test_examples, test_labels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proto_pattern",
        default=None,
        help="File pattern of QuestionCandidate protos.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help=("Path of folder to save model to."),
    )
    parser.add_argument(
        "--model_hparams",
        type=str,
        default="",
        help=(
            "Comma separated list of 'name=value' pairs."
            "Parameters are learning_rate, hidden_size, encoding_dim."
        ),
    )
    args = parser.parse_args()
    hparams = _DEFAULT_HPARAMS.parse(args.model_hparams)
    train_model(hparams, args.proto_pattern, args.output_path)

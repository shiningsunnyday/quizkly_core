import os
from os import path
import shutil
import tempfile

import tensorflow as tf
from gensim.models.word2vec import Word2Vec

from models import distractor_encoder, sentence_classifier
from models.binary_gap_classifier import train_model
from models.distractor_encoder import HParams as DistHParams
from models.sentence_classifier import HParams as SentHParams


def save_model(hparams, model, saved_model_dir):
    test_dir = tempfile.mkdtemp()
    run_config = tf.estimator.RunConfig(model_dir=test_dir)
    estimator = tf.estimator.Estimator(
        config=run_config,
        model_fn=model.model_fn,  # First-class function
        params=hparams,  # HParams
    )
    estimator.train(
        model.input_fn(hparams, tf.estimator.ModeKeys.TRAIN), steps=1
    )
    exported_path = estimator.export_savedmodel(
        export_dir_base=saved_model_dir,
        serving_input_receiver_fn=model.input_fn(
            hparams, tf.estimator.ModeKeys.PREDICT
        ),
    )
    new_path = b"/".join(exported_path.split(b"/")[:-1])
    new_path = os.path.join(new_path, b"saved_model")
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.rename(exported_path, new_path)
    shutil.rmtree(test_dir)


def save_sentence_classifier():
    hparams = SentHParams(
        train_records="datasets/testdata/squad_test.tfrecords",
        train_batch_size=2,
        sentence_feature="sentence",
        context_feature="context",
        label_feature="question_worthy",
        hidden_size=2,
        num_classes=2,
        learning_rate=0.02,
        is_test=True,
    )
    saved_model_dir = path.join("models", "test_data", "sentence_classifier")
    save_model(hparams, sentence_classifier, saved_model_dir)


def save_binary_gap_classifier():
    hparams = tf.contrib.training.HParams(
        learning_rate=0.9, hidden_size=100, encoding_dim=3
    )
    saved_model_path = path.join("models", "test_data", "gap_classifier.h5")
    train_model(
        hparams, "datasets/testdata/question_candidates_test", saved_model_path
    )


def save_word2vec_model():
    sentences = ["Quizkly automatically generates questions.".split(),
                 "A water bottle contains water".split(),
                 "John went to the market.".split(),
                 " The market was huge.".split()]
    params = {
        'size': 3,
        'window': 2,
        'min_count': 1,
        'sample': 1E-5,
    }
    word2vec = Word2Vec(sentences, **params)
    saved_model_dir = path.join("models", "test_data", "word_model")
    word2vec.wv.save_word2vec_format(saved_model_dir)


def save_distractor_encoder():
    hparams = DistHParams(
        train_records="datasets/testdata/distractors_test",
        eval_records="datasets/testdata/distractors_test",
        question_feature="question",
        answer_feature="answer",
        distractor_feature="distractor",
        negatives_feature="negatives",
        negatives_length_feature="negatives_length",
        vector_size=10,
        train_batch_size=5,
        eval_batch_size=5,
        learning_rate=0.001,
        is_test=True,
    )
    saved_model_dir = path.join("models", "test_data", "distractor_encoder")
    save_model(hparams, distractor_encoder, saved_model_dir)


if __name__ == "__main__":
    save_sentence_classifier()
    save_binary_gap_classifier()
    save_word2vec_model()
    save_distractor_encoder()

import os
from os import path
import shutil
import tempfile

import tensorflow as tf

from models import sentence_classifier
from models.sentence_classifier import HParams


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
    hparams = HParams(
        train_records="datasets/testdata/*.tfrecords",
        train_batch_size=2,
        sentence_feature="sentence",
        label_feature="question_worthy",
        hidden_size=2,
        num_classes=2,
        is_test=True,
    )
    saved_model_dir = path.join("models", "test_data", "sentence_classifier")
    save_model(hparams, sentence_classifier, saved_model_dir)


if __name__ == "__main__":
    save_sentence_classifier()

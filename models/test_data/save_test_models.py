from os import path

import tensorflow as tf
import tensorflow_hub as hub

from models.base_model import Mode
from models.sentence_classifier import SentenceClassifier, HParams

def save_sentence_classifier():
    hparams = HParams(
        train_records="dummy",
        train_batch_size=2,
        sentence_feature="sentence",
        label_feature="question_worthy",
        hidden_size=2,
        num_classes=2
    )
    saved_model = path.join("models", "test_data",
        "sentence_classifier", "saved_model")
    checkpoint = path.join("models", "test_data",
        "sentence_classifier", "checkpoint.ckpt")
    with tf.Session(graph=tf.Graph()) as sess:
        hub_module = hub.Module(
            "https://tfhub.dev/google/universal-sentence-encoder/1")
        train_model = SentenceClassifier(hparams, Mode.TRAIN, hub_module)
        train_model.build_model()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        train_model.saver.save(sess, checkpoint)
        tf.logging.info("Wrote %s", checkpoint)
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model)
    SentenceClassifier.export_model(hparams, checkpoint, builder)

if __name__ == "__main__":
    save_sentence_classifier()
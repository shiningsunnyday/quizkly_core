"""Tensorflow utility functions for training"""

import os

from tqdm import trange
import tensorflow as tf

from trainer.utils import save_dict_to_json
from trainer.evaluation import evaluate_sess


def train_sess(sess, model, num_steps, writer):
    """Train the model on `num_steps` batches

    Args:
        sess: current session
        model: (model) model to be trained.
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries
    """
    # Get relevant graph operations or nodes needed for training
    loss = model.loss
    train_op = model.train_op
    summary_op = model.summary_op
    global_step = tf.train.get_global_step()

    # Use tqdm for progress bar
    t = trange(num_steps)
    for i in t:
        # Evaluate summaries for tensorboard only once in a while
        if i % 5 == 0:
            # Perform a mini-batch update
            _, loss_val, summ, global_step_val = sess.run(
                [train_op, loss,
                 summary_op, global_step])
            # Write summaries for tensorboard
            writer.add_summary(summ, global_step_val)
        else:
            _, loss_val = sess.run([train_op, loss])
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss_val))


def train_and_evaluate(train_model, eval_model, model_dir, num_epochs,
                       eval_steps, tracking_metric="accuracy",
                       restore_from=None):
    """Train the model and evaluate every epoch.

    Args:
        train_model_spec: (model) model to be trained.
        eval_model_spec: (model) model graph to be evaluated.
        model_dir: (string) directory containing config, weights and log
        num_epochs: (int) number of training epochs.
        restore_from: (string) directory or file containing weights to
            restore the graph.
        tracking_metric: (string) name of metric to use to save best
            performing models.
    """
    # Initialize tf.Saver instances to save weights during training.
    last_saver = tf.train.Saver()  # will keep last 5 epochs.
    # only keep 1 best checkpoint (best on eval).
    best_saver = tf.train.Saver(max_to_keep=1)
    begin_at_epoch = 0

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model.variable_init_op)

        # Reload weights from directory if specified
        if restore_from is not None:
            tf.logging.info(
                "Restoring parameters from {}".format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(
            os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(
            os.path.join(model_dir, 'eval_summaries'), sess.graph)

        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch,
                           begin_at_epoch + num_epochs):
            # Run one epoch
            tf.logging.info("Epoch {}/{}".format(
                epoch + 1, begin_at_epoch + num_epochs))
            # One batch in one epoch (one full pass over the training set)
            train_sess(sess, train_model, 10, train_writer)

            # Save weights
            last_save_path = os.path.join(
                model_dir, 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Evaluate for one epoch on validation set
            metrics = evaluate_sess(
                sess, eval_model, eval_steps, eval_writer)

            # If best_eval, best_save_path
            eval_acc = metrics[tracking_metric]
            if eval_acc > best_eval_acc:
                # Store new best accuracy
                best_eval_acc = eval_acc
                # Save weights
                best_save_path = os.path.join(
                    model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(
                    sess, best_save_path, global_step=epoch + 1)
                tf.logging.info(
                    "- Found new best {}, saving in {}".format(
                        tracking_metric, best_save_path))
                # Save best eval metrics in a json file in the model directory
                best_json_path = os.path.join(
                    model_dir, "metrics_eval_best_weights.json")
                save_dict_to_json(metrics, best_json_path)

            # Save latest eval metrics in a json file in the model directory
            last_json_path = os.path.join(
                model_dir, "metrics_eval_last_weights.json")
            save_dict_to_json(metrics, last_json_path)

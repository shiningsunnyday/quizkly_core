"""Train the model"""

import argparse
import importlib
import os

import tensorflow as tf
import tensorflow_hub as hub

from models.base_model import Mode
from trainer.training import train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--model_class', required=True,
                    help="Class name of model")
parser.add_argument('--hparams_class', required=True,
                    help="Hparams name of model")
parser.add_argument('--num_eval_steps', default=10, type=int,
                    help="Number of eval batches.")
parser.add_argument('--num_epochs', default=1000, type=int,
                    help="Number of training epochs.")
parser.add_argument('--tracking_metric', default='accuracy',
                    help="Metric to gauge best model to save.")
parser.add_argument('--model_dir', required=True)
parser.add_argument(
    '--restore_dir', default=None,
    help="Optional, directory containing weights to reload before training")
parser.add_argument(
    '--hub_module_url', default=None,
    help="URL to hub module if model requires it.")


def get_attribute_from_path(path):
    """Get named attribute from path.

    Args:
        path: full path to an attribute

    Returns: function or class or object matching path
             or None if path is invalid"""
    paths = path.rsplit('.', 1)
    if len(paths) != 2:
        return None

    module = importlib.import_module(paths[0])
    return getattr(module, paths[1])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    # Check that we are not overwriting some previous experiment
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwriting = (model_dir_has_best_weights and
                   args.restore_dir is None)
    assert not overwriting, \
        "Weights found in model_dir, aborting to avoid overwrite"

    # Define the models.
    # 2 different set of nodes that share weights for train and eval.
    tf.logging.info("Creating the model...")
    _Model = get_attribute_from_path(args.model_class)
    hparams = get_attribute_from_path(args.hparams_class)
    hub_module = None
    if args.hub_module_url:
        hub_module = hub.Module(args.hub_module_url, trainable=True)
    train_model = _Model#(hparams, Mode.TRAIN, hub_module=hub_module)
    eval_model = None
    # eval_model = _Model(hparams, Mode.EVAL, hub_module=hub_module)
    #with tf.variable_scope('model'):
    #train_model.build_model()
    # with tf.variable_scope('model', reuse=True):
    #     eval_model.build_model()
    tf.logging.info("- done.")

    # Train the model
    tf.logging.info(
        "Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(
        train_model, eval_model, _Model, hparams, args.model_dir,
        args.num_epochs, args.num_eval_steps,
        tracking_metric=args.tracking_metric,
        restore_from=args.restore_dir)

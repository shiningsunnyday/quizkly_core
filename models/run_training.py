"""Train the model"""

import argparse
import importlib
import os

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model_class", required=True, help="Class name of model")
parser.add_argument(
    "--hparams_class", required=True, help="Hparams name of model"
)
parser.add_argument(
    "--num_eval_steps", default=10, type=int, help="Number of eval batches."
)
parser.add_argument(
    "--num_epochs", default=1000, type=int, help="Number of training epochs."
)
parser.add_argument(
    "--save_summary_steps",
    default=None,
    type=int,
    help="Save summaries every this many steps.",
)
parser.add_argument(
    "--save_checkpoints_steps",
    default=None,
    type=int,
    help=(
        "Save checkpoints every this many steps. "
        "Can not be specified with save_checkpoints_secs."
    ),
)
parser.add_argument(
    "--save_checkpoints_secs",
    default=240,
    type=int,
    help=(
        "Save checkpoints every this many seconds. "
        "Can not be specified with save_checkpoints_steps."
    ),
)
parser.add_argument("--model_dir", required=True)
parser.add_argument(
    "--restore_dir",
    default=None,
    help="Optional, directory containing weights to reload before training",
)


def get_attribute_from_path(path):
    """Get named attribute from path.

    Args:
        path: full path to an attribute

    Returns: function or class or object matching path
             or None if path is invalid"""
    paths = path.rsplit(".", 1)
    if len(paths) != 2:
        return None

    module = importlib.import_module(paths[0])
    return getattr(module, paths[1])


def train_eval_model(
    hparams,
    model_fn,
    input_fn,
    run_config,
    train_steps=1000,
    eval_steps=10,
    eval_throttle_secs=240,
    warm_start_from=None,
):
    """Function to train and evaluate a tf.estimator.Estimator

    Args:
        hparams: dict or tf.contrib.HParams containing parameters for model.
        model_fn: model_fn for estimator.
        input_fn: input_fn for estimator.
        run_config: tf.estimator.RunConfig containting parameters for training.
        train_steps: number of training steps.
        eval_steps: number of batches to run eval on.
        eval_throttle_secs: how often to run eval.
        warm_start_from: path to checkpoint to start training from, if any.
    """
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=hparams,  # HParams
        config=run_config,  # RunConfig,
        warm_start_from=warm_start_from,  # Checkpoint to start from.
    )
    train_input_fn = input_fn(hparams, tf.estimator.ModeKeys.TRAIN)
    eval_input_fn = input_fn(hparams, tf.estimator.ModeKeys.TRAIN)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=eval_steps,
        start_delay_secs=30,
        throttle_secs=eval_throttle_secs,
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.set_random_seed(230)
    args = parser.parse_args()

    warm_start_from = None
    if os.path.isdir(args.model_dir):
        warm_start_from = args.model_dir

    if args.restore_dir and os.path.isdir(args.restore_dir):
        warm_start_from = args.restore_dir

    run_config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_summary_steps=args.save_summary_steps,
        save_checkpoints_steps=args.save_checkpoints_steps,
        save_checkpoints_secs=args.save_checkpoints_secs,
    )

    _Model = importlib.import_module(args.model_class)
    hparams = get_attribute_from_path(args.hparams_class)
    estimator = train_eval_model(
        hparams,
        _Model.model_fn,
        _Model.input_fn,
        run_config,
        train_steps=args.num_epochs,
        eval_steps=args.num_eval_steps,
        eval_throttle_secs=args.save_checkpoints_secs or 240,
        warm_start_from=warm_start_from,
    )
    estimator.export_savedmodel(
        export_dir_base=args.model_dir,
        serving_input_receiver_fn=_Model.input_fn(
            hparams, tf.estimator.ModeKeys.PREDICT
        ),
    )

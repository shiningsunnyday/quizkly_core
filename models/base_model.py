"""Base Tensorflow Model Class"""
from enum import Enum


class Mode(Enum):
    """Enum class """

    TRAIN = 1
    EVAL = 2
    INFERENCE = 3


class BaseModel(object):
    """
    Base class for all tensorflow models.

    Attributes:
        variable_init_op: op to initialize variables.
        iterator_init_op: op to initialize dataset iterator.
        loss: loss scalar.
        metrics_init_op: op to initialize eval metrics, if mode is
            EVAL.
        metrics: Dictionary containing a mapping from a given metric
            name (string) to a (value tensor, update op) tuple, if
            mode is EVAL.
        summary_op: op for tensorboard summary.
        train_op: op for training, if mode is TRAIN.
    """

    def __init__(self, hparams, mode, hub_module=None):
        """
        Creates a model.

        Args:
            hparams: tensorflow hparams object specifying hyperparamaters.
            mode: mode enum
            hub_module: optional tensorflow_hub module, if required in graph.
        """
        self._hparams = hparams
        self._mode = mode
        self._hub_module = hub_module

    def build_model(self):
        """
        Builds tensorflow graph.
        """
        raise NotImplementedError

    def save_model(self, builder, session):
        """Saves the model using the given builder.

        Saving inference model.

        Args:
            builder: a SavedModelBuilder object to save the model.
            session: the session in which the model was built.
        """
        raise NotImplementedError

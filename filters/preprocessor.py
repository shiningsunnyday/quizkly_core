""" Functions for preprocessing texts before pipeline is run. """
import re


def preprocess_text(text):
    """ Preprocesses text before pipeline is run.
    Args:
        text: string to preprocess.

    Returns:
        preprocessed string.
    """
    text = _strip_bracket_text(text)
    text = _remove_extra_whitespace(text)
    return text


def _strip_bracket_text(text, nested_level=2):
    """ Remove brackets and texts within from text.
    Args:
        text: text to strip brackets from.
        nested_level: level of nesting to strip.

    Returns:
        processed text.
    """
    for i in range(nested_level):
        text = re.sub(r"(\([^()]*\)|\[[^[]]*\])", r"", text)
    return text


def _remove_extra_whitespace(text):
    """ Removes extra whitespace. """
    return re.sub(' +', ' ', text)

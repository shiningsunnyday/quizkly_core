"""Utility functions for models """

import tensorflow as tf


def hub_embedding_lookup_unique(hub_module, tokens, name=None):
    """Embedding lookup that avoids duplicate lookups for hub_modules.
    This can save time in the case of repeated tokens.
    Same interface as embedding_lookup. Except it supports
    multi-dimensional `tokens` which allows to not reshape input/output to
    fit gather.
    Args:
        hub_module: hub_module that embeds 1d string tensors.
        tokens: A tensor of shape [d1, d2...], with type `string` to be
            encoded by `hub_module`.
        name: A name for this operation (optional).
    Returns:
        A `Tensor` of shape [d1, d2, .., d] where d is the embedding dimension.
    """
    with tf.name_scope(name, "EmbeddingLookupUnique", [tokens]):
        tokens = tf.convert_to_tensor(tokens)
        shape = tf.shape(tokens)
        tokens_flat = tf.reshape(tokens, tf.reduce_prod(shape, keepdims=True))
        unique_tokens, idx = tf.unique(tokens_flat)
        unique_embeddings = hub_module(unique_tokens)
        embeds_flat = tf.gather(unique_embeddings, idx)
        embed_shape = tf.concat([shape, tf.shape(unique_embeddings)[1:]], 0)
        embeds = tf.reshape(embeds_flat, embed_shape)
        embeds.set_shape(
            tokens.get_shape().concatenate(unique_embeddings.get_shape()[1:])
        )
        return embeds

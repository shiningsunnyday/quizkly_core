""" Client for retrieving elmo encodings"""
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


_DEFAULT_URL = "https://tfhub.dev/google/elmo/2"


class ElmoClient(object):
    def __init__(self, hub_url=_DEFAULT_URL):
        self._graph = tf.Graph()
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._input_texts = tf.placeholder(tf.string, shape=[None])
            self._input_lengths = tf.placeholder(tf.int32, shape=[None])
            sparse_words = tf.string_split(self._input_texts, skip_empty=False)
            words = tf.sparse_tensor_to_dense(
                sparse_words, default_value="<UNK>"
            )
            self._elmo_module = hub.Module(hub_url)
            self._sequence_embeddings = self._elmo_module(
                inputs={"tokens": words, "sequence_len": self._input_lengths},
                signature="tokens",
                as_dict=True,
            )["elmo"]
            self._session.run(tf.global_variables_initializer())
            self._session.run(tf.tables_initializer())

    def encode(self, texts):
        """ Returns a numpy array of encodings
        Args:
            texts: list of strings
        """
        input_lengths = [len(text.split(" ")) for text in texts]
        return self._session.run(
            self._sequence_embeddings,
            {self._input_texts: texts, self._input_lengths: input_lengths},
        )

    def get_sliced_encodings(self, texts, slice_ids):
        """ Returns a numpy array of encodings summed across slice_ids
        Args:
            texts: list of strings
            slice_ids: list of tuples containing start & end indices.
        """
        embeddings = self.encode(texts)
        sliced_embeddings = []
        for i, embedding in enumerate(embeddings):
            sliced_embedding = np.sum(
                embedding[slice_ids[i][0]: slice_ids[i][1], :], axis=0
            )
            sliced_embedding /= math.sqrt(
                slice_ids[i][1] - slice_ids[i][0] + 1
            )
            sliced_embeddings.append(sliced_embedding)
        return sliced_embeddings

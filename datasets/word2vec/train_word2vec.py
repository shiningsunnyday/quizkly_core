""" Script to train word vectors on wikipedia data with gensim.
Usage:
    python -m service.train_word2vec \
        --wiki_dump local/wiki_dump
        --temp_text_file temp_file
        --out_path model.word2vec
        --iterations 10
"""
import argparse
import multiprocessing

import gensim
from gensim.models.word2vec import LineSentence, Word2Vec

import nltk

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument(
    "--wiki_dump",
    required=True,
    help="Location of wikipdia dump.",
)
_PARSER.add_argument(
    "--temp_text_file",
    required=True,
    help="Location of temporary text file with all sentences.",
)
_PARSER.add_argument(
    "--out_path",
    required=True,
    help="Location to output word2vec model.",
)
_PARSER.add_argument(
    "--iterations",
    default=10,
    help="Number of iterations.",
    type=int
)
_FLAGS = _PARSER.parse_args()


def _main():
    wiki = gensim.corpora.WikiCorpus(_FLAGS.wiki_dump, lemmatize=False)
    with open(_FLAGS.temp_text_file, "w") as output:
        for i, text in enumerate(wiki.get_texts()):
            sentences = nltk.sentence_tokenize(text)
            if len(sentences) < 10:
                continue
            for s in sentences:
                output.write(s + "\n")
    sentences = LineSentence(_FLAGS.temp_text_file)
    bigram_transformer = gensim.models.phrases.Phrases(sentences)
    trigram_transformer = gensim.models.Phrases(bigram_transformer[sentences])
    train_sentences = gensim.utils.RepeatCorpusNTimes(
        trigram_transformer[sentences], _FLAGS.iterations)
    params = {
        'size': 100,
        'window': 10,
        'min_count': 10,
        'workers': max(1, multiprocessing.cpu_count() - 1),
        'sample': 1E-5,
    }
    word2vec = Word2Vec(train_sentences, **params)
    word2vec.save(_FLAGS.out_path)


if __name__ == "__main__":
    _main()

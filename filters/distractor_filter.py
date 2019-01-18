""" Functions to choose distractors for a chosen gap."""
import collections
import itertools
import re
import string
from operator import itemgetter

from nltk import stem
from nltk.corpus import wordnet as wn
from spacy.lang.en.stop_words import STOP_WORDS

from proto import question_candidate_pb2

# Converts spacy part of speech tags to wordnet's format.
POS_TO_WN = collections.defaultdict(
    lambda: wn.NOUN,
    {"ADV": wn.ADV, "VERB": wn.VERB,
    "ADJ": wn.ADJ, "NOUN": wn.NOUN,
    "PROPN": wn.NOUN
})
PUNC_REGEX = re.compile('[%s]' % re.escape(string.punctuation))


def filter_distractors(question_candidates, spacy_docs, parser,
                       word_model, num_dists=4):
    """
    Finds distractors from given word2vec model for question candidates.

    Args:
        question_candidate: QuestionCandidate proto
        spacy_doc: spacy doc
        parser: spacy parser
        word_model: word2vec model
        num_dists: number of distractors
    Returns:
        list of distractor strings.
    """

    stemmer = stem.snowball.SnowballStemmer("english")
    return [
        filter_distractors_single(qc, spacy_docs[i], parser, word_model,
                                  stemmer, num_dists)
        for i, qc in enumerate(question_candidates)
    ]


def filter_distractors_single(question_candidate, spacy_doc, parser,
                              word_model, stemmer, num_dists=4):
    """
    Finds distractors from given word2vec model for one question_candidate.

    Args:
        question_candidate: QuestionCandidate proto
        spacy_doc: spacy doc
        parser: spacy parser
        word_model: word2vec model
        stemmer: nltk stemmer
        num_dists: number of distractors
    Returns:
        list of distractor strings.
    """
    gap = question_candidate.gap
    question = question_candidate.question_sentence
    distractors = nearest_neighbors(gap, word_model, num_dists * 10)
    if not distractors:
        return []
    distractors = filter_words_in_sent(question, distractors, stemmer)
    distractors = filter_stopword(distractors)
    distractors = filter_part_of_speech(gap, distractors, parser)
    distractors = filter_wordnet(gap, distractors, stemmer)
    distractors = rescore(word_model, spacy_doc, distractors, gap)
    final_distractors = [d[0] for d in distractors[:num_dists]]
    final_distractors = post_process(
        final_distractors, gap, spacy_doc, stemmer)
    question_candidate.distractors.extend(
        [question_candidate_pb2.Distractor(text=dist)
         for dist in final_distractors]
    )
    return final_distractors


def nearest_neighbors(gap, word_model, topn=20):
    """
    Obtains words closest to the gap in the provided word model vector space.

    Args:
        gap: gap to find nearest neighbors for.
        word_model: gensim.models.Word2Vec word vector space.
        topn: number of nearest neighbors to get.

    Returns:
        list of (neighbor, score) tuples where score is the cosine sim between
        gap and the neighbor.
    """
    # n-grams are separated by underscores in the word model.
    phrase = gap.text.replace(' ', '_')
    neighbors = []

    try:
        neighbors.extend(word_model.most_similar_cosmul(
            positive=[phrase], topn=topn))
    except KeyError:
        pass

    # Try to get more candidates using lowercase.
    # check that second char is lower-case to account
    # for terms like DNA, etc.
    if phrase[0].isupper() and phrase[1].islower():
        phrase_low = phrase.lower()
        try:
            neighbors.extend(
            word_model.most_similar_cosmul(
                positive=[phrase_low], topn=topn)
            )
            neighbors.sort(key=lambda x: x[1], reverse=True)
            neighbors = neighbors[:topn]
        except KeyError:
            pass

    # change underscores back to spaces and remove punctuation.
    neighbors = [[n[0].replace("_", " "), n[1]] for n in neighbors]
    punc_stripper = str.maketrans('', '', string.punctuation)
    neighbors = [
        [n[0].translate(punc_stripper), n[1]]
        for n in neighbors
    ]
    # remove empty candidates that might have just been punctuations/spaces.
    candidates = [[n[0], n[1]] for n in neighbors if n[0].strip() != '']
    return candidates


def _stem_words(stemmer, words):
    """ Stems list of words. """
    return [stemmer.stem(word.lower()) for word in words]


def filter_words_in_sent(sentence, distractors, stemmer):
    """
    Removes distractor candidates that appear in sentence.
    Comparison is done on the stemmed tokens.

    Args:
        distractors: list of (candidate, score) tuples
        sentence: question sentence
        stemmer: nltk stemmer.

    Returns:
        list of (candidate, score) tuples
    """

    # remove puncs from question
    punc_stripper = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(punc_stripper)
    stemmed_sentence = _stem_words(stemmer, sentence.split())
    filtered_distractors = []

    for pair in distractors:
        stemmed_phrase = _stem_words(stemmer, pair[0].split(" "))
        if not all(w in stemmed_sentence for w in stemmed_phrase):
            filtered_distractors.append(pair)

    return filtered_distractors


def filter_part_of_speech(gap, distractors, parser):
    """
    Removes distractor candidates that have a different
    part of speech from selected gap.

    Args:
        gap: chosen gap.
        distractors: list of (candidate, score) tuples.
        parser: spacy parser

    Returns:
        list of (candidate, score) tuples.
    """

    ref_tag = gap.pos_tags[-1]

    filtered_distractors = []
    phrases = [pair[0] for pair in distractors]
    tokenized = parser.tokenizer.pipe(phrases, batch_size=len(phrases),
                                      n_threads=4)
    tagged_phrases = parser.tagger.pipe(tokenized, batch_size=len(phrases),
                                        n_threads=4)
    for dist in distractors:
        tagged = next(tagged_phrases)
        dist_tag = tagged[-1].pos_
        if ref_tag == dist_tag:
            filtered_distractors.append(dist)
    return filtered_distractors


def filter_stopword(distractors):
    """
    Remove stopwords from distractors.

    Args:
        distractors: list of (candidate, score) tuples
        sentence: question sentence

    Returns:
        list of (candidate, score) tuples
    """
    filtered_distractors = []
    for pair in distractors:
        phrase = pair[0].split(" ")
        phrase = [w for w in phrase if w not in STOP_WORDS]
        if len(phrase) <= 0:
            continue
        filtered_distractors.append([" ".join(phrase), pair[1]])
    return filtered_distractors


def filter_wordnet(gap, distractors, stemmer):
    """
    Filter by wordnet relations.

    Args:
        gap: chosen gap.
        distractors: list of (candidate, score) tuples
        stemmer: nltk stemmer.

    Returns:
        list of (candidate, score) tuples
    """
    # preparing wordnet synsets for subsequent filters
    candidates_syn, gap_syn, gap_hypomeronyms = _prep_wordnet_synsets(
        gap, distractors)

    candidates = zip(candidates_syn, distractors)

    # filtering by wordnet similarity
    candidates = filter_wordnetsim(candidates, gap_syn)
    # filtering meronyms and hyponyms
    candidates = filter_hypomeronyms(
        candidates, gap, gap_syn, gap_hypomeronyms, stemmer)
    candidates = filter_duplicates(candidates, gap, gap_syn, stemmer)
    return list(zip(*candidates))[1]


def _prep_wordnet_synsets(gap, distractors):
    """
    Queries wordnet synsets for gap and distractors
    """
    ref_tag = gap.pos_tags[-1]
    gap_syn = wn.synsets(gap.text.replace(' ', '_'), POS_TO_WN[ref_tag])
    gap_hypomeronyms = []
    candidates_syn = []
    for syn in gap_syn:
        gap_hypomeronyms += _get_hypomeronyms(syn)
    for cand, _ in distractors:
        candidates_syn.append(
            wn.synsets(cand.replace(" ", "_"), POS_TO_WN[ref_tag]))
    return candidates_syn, gap_syn, gap_hypomeronyms


def _get_hypomeronyms(syn):
    """
    Return list of mero, hypo, holo, similar_tos for a certain synset.
    """
    hypomeronyms = []
    hypomeronyms += [i for i in syn.closure(lambda s: s.hyponyms())]
    hypomeronyms += [i for i in syn.closure(lambda s: s.part_meronyms())]
    hypomeronyms += [i for i in syn.closure(lambda s: s.member_holonyms())]
    hypomeronyms += syn.similar_tos()
    return hypomeronyms


def wordnet_sim(set_a, set_b):
    """
    Computes maximum similarity between two synsets (s,t) spanning multiple
    senses of two different words.

    Args:
        set_a: wordnet synset
        set_b: wordnet synset

    Returns:
        float containing max similarity of all possible pairings
        between elements in the synsets.
    """
    # permutate all possible sim calcs
    possible_pairs = itertools.product(set_a, set_b)
    scores = []
    for pair in possible_pairs:
        score = pair[0].path_similarity(pair[1])
        if score is not None:
            scores.append(score)
    if scores:
        return max(scores)
    else:
        return 0.1


def filter_wordnetsim(distractors, gap_syn, thres=0.1):
    """
    Removes distractors which are too far away from the gap in WordNet.

    Args:
        distractors: list of tuples of form (wn synset, (candidate, score)).
        gap_syn: gap synset.
        thres: simiilarity threshold (keep if above).

    Returns:
        list of tuples of form (wn synset, (candidate, score)).
    """
    filtered_distractors = []
    for dist in distractors:
        dist_syn = dist[0]
        if wordnet_sim(dist_syn, gap_syn) >= thres:
            filtered_distractors.append(dist)
    return filtered_distractors


def _check_hypomeronym(gap, dist, gap_hypomeronyms, stemmer):
    """
    Check if the dist is a hyponym or meronym of gap.

    Args:
        gap: (string, synset) tuple of gap
        dist: (string, sysnset) tuple of distractor
        gap_hypomeronym: hyponyms and meronyms of the gap from wordnet.
        stemmer: nltk stemmer

    Returns:
        boolean.
    """
    g_str, g_syn, d_str, d_syn = gap[0], gap[1], dist[0], dist[1]
    g_str = PUNC_REGEX.sub(' ', g_str)
    d_str = PUNC_REGEX.sub(' ', d_str)
    # check if the distractor is containing in the gap.
    # e.g. blood cell is in red blood cell.
    if all([w in d_str.lower() for w in g_str.lower().split(' ')]):
        return True

    d_str_stemmed = _stem_words(stemmer, d_str.lower().split(' '))
    g_str_stemmed = _stem_words(stemmer, g_str.lower().split(' '))
    if all([w in d_str_stemmed for w in g_str_stemmed]):
        return True

    if not d_syn or not g_syn:
        return False

    return not set(d_syn).isdisjoint(set(gap_hypomeronyms))


def filter_hypomeronyms(distractors, gap, gap_syn, gap_hypomeronyms, stemmer):
    """
    Removes distractors that are hyponyms of the gap.

    Args:
        distractors: list of tuples of form (wn synset, (candidate, score)).
        gap: gap.
        gap_syn: gap synset.
        gap_hypomeronyms: list of mero, hypo, holo, similar_tos for gap_syn.
        stemmer: nltk stemmer.

    Returns:
        list of tuples of form (wn synset, (candidate, score)).
    """
    filtered_distractors = []
    for dist in distractors:
        dist_syn, (dist_str, _) = dist
        gap_str, gap_syn = gap.text, gap_syn

        same_last_word = (
            _stem_words(stemmer, gap_str.split(' ')[-1:]) ==
            _stem_words(stemmer, dist_str.split(' ')[-1:])
        )
        if (len(gap_str.split(' ')) > 1 and
                len(dist_str.split(' ')) > 1 and same_last_word):
            dist_str = ' '.join(dist_str.split(' ')[:-1])
            gap_str = ' '.join(gap_str.split(' ')[:-1])
            ref_tag = gap.pos_tags[-2]
            dist_syn = wn.synsets(
                dist_str.replace(" ", "_"), POS_TO_WN[ref_tag])
            gap_syn = wn.synsets(gap_str.replace(" ", "_"), POS_TO_WN[ref_tag])
            gap_hypomeronyms_first = []
            for syn in gap_syn:
                gap_hypomeronyms_first += _get_hypomeronyms(syn)
            if _check_hypomeronym((gap_str, gap_syn), (dist_str, dist_syn),
                                  gap_hypomeronyms_first, stemmer):
                continue

        if not _check_hypomeronym((gap_str, gap_syn), (dist_str, dist_syn),
                                  gap_hypomeronyms, stemmer):
            filtered_distractors.append(dist)
    return filtered_distractors


def filter_duplicates(distractors, gap, gap_syn, stemmer):
    """
    Remove distractors which are duplicates of gaps/other distractors
    """
    filtered_distractors = []
    delete_idx = set([])
    for i, dist in enumerate(distractors):
        if i in delete_idx:
            continue

        dist_syn, (dist_str, dist_score) = dist
        gap_str, gap_syn = gap.text, gap_syn

        if check_duplication((dist_str, dist_syn),
                             (gap_str, gap_syn), stemmer):
            delete_idx.add(i)
            continue

        for j, dist_check in enumerate(distractors):
            if i == j or j in delete_idx:
                continue
            check_syn, (check_str, check_score) = dist_check
            if check_duplication((dist_str, dist_syn),
                                 (check_str, check_syn), stemmer):
                # delete the one with the lower wordvec score
                if dist_score > check_score:
                    delete_idx.add(j)
                else:
                    delete_idx.add(i)
                break

    filtered_distractors = [
        dist for i, dist in enumerate(distractors) if i not in delete_idx
    ]
    return filtered_distractors


def check_duplication(word_x, word_y, stemmer):
    """
    Checks duplication by stemming and WordNet.

    Args:
        word_x: (string, synset tuple)
        word_y: (string, sysnset tuple)

    Returns:
        boolean indicating whether of not the words are duplicates.
    """
    x_str, x_sn, y_str, y_sn = word_x[0], word_x[1], word_y[0], word_y[1]
    x_str = PUNC_REGEX.sub(' ', x_str)
    y_str = PUNC_REGEX.sub(' ', y_str)
    same_word = (_stem_words(stemmer, x_str.lower().split(' ')) ==
                 _stem_words(stemmer, y_str.lower().split(' ')))

    if same_word:
        return True

    if x_sn and y_sn:  # only compare if word has a synset in wordnet
        same_synset = not set(x_sn).isdisjoint(set(y_sn))
    else:
        same_synset = False

    # TODO: add wikipedia trie functionality @girish
    return same_synset


def _dice(word_a, word_b):
    """
    Returns dice coefficient between two words
    """
    len_a = len(word_a)
    len_b = len(word_b)
    a_chars = set(word_a)
    b_chars = set(word_b)
    overlap = len(a_chars & b_chars)
    return 2.0 * overlap / (len_a + len_b)


def rescore(word_model, spacy_doc, distractors, gap):
    """
    Rescoring by context score and lexical similarity.

    Args:
        word_model: word2vec model
        spacy_doc: spacy doc of question sentence.
        distractors: list of tuples of form (candidate, score).
        gap: gap to compare to.
    """
    context_scores = _get_context_scores(
        word_model, spacy_doc, gap, distractors)
    lex_sim = [_dice(d, gap.text) for d, _ in distractors]

    for i, _ in enumerate(distractors):
        distractors[i][1] = (1.6/6 * distractors[i][1] +
                             3.2/6 * context_scores[i] +
                             1.2/6 * lex_sim[i])

    distractors = sorted(distractors, key=itemgetter(1), reverse=True)
    return distractors


def _get_context_scores(word_model, spacy_doc, gap, distractors):
    """
    Computes wordvec similarity with between the word vector of the candidate
    and other words in the sentence.
    """
    gap_phrase = gap.text.replace(' ', '_')

    context_scores = []
    for candidate in distractors:
        context_score = 0.0
        candidate_str = candidate[0].replace(" ", "_")
        for token in spacy_doc:
            if token.i < gap.end_index and token.i >= gap.start_index:
                continue
            if token.is_stop or token.is_punct:
                continue
            try:
                sim = word_model.similarity(gap_phrase, candidate_str)
                positional_weight = 0
                if token.i < gap.start_index:
                    positional_weight = 1.0/(gap.start_index - token.i)
                else:
                    positional_weight = 1.0/(token.i - gap.end_index + 1)
                context_score += positional_weight * sim
            except KeyError:
                context_score += 0
        context_scores.append(context_score)

    return context_scores


def post_process(distractors, gap, spacy_doc, stemmer):
    """
    Deals with the capitalising, pos-transformation of gaps
    """
    prev_token, after_token = None, None
    if gap.start_index > 1:
        prev_token = spacy_doc[gap.start_index - 1]
    if gap.end_index < len(spacy_doc):
        after_token = spacy_doc[gap.end_index]

    for i, distractor in enumerate(distractors):
        # ensure capitalization same as gap-phrase
        if gap.text[0].isupper():
            distractors[i] = distractor.capitalize()
        distractor_words = distractor.split(' ')
        if (prev_token and
                _stem_words(stemmer, distractor_words[0:1]) ==
                _stem_words(stemmer, [prev_token.orth_])):
            distractors[i] = ' '.join(distractor_words[1:])
        if (after_token and
                _stem_words(stemmer, distractor_words[-1:]) ==
                _stem_words(stemmer, [after_token.orth_])):
            distractors[i] = ' '.join(distractor_words[:-1])
    return distractors

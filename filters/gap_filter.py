""" Functions to filter gaps from candidate question sentence.
"""
import math
import logging

import numpy

from proto import question_candidate_pb2


def filter_gaps(spacy_docs, batch_size=50, elmo_client=None):
    """ Generates gap candidates for a list of sentences.

    Args:
        spacy_docs: list of Spacy doc objects.
        batch_size: batches of sentences to process at once.
        elmo_client: ElMo client to encode gaps. If None, gaps are not encoded.

    Returns:
        A generator of lists containing QuestionCandidate protos, of length
        batch_size.
    """

    def _encode_gaps(question_candidates, elmo_client):
        sents = [cand.question_sentence for cand in question_candidates]
        encodings = elmo_client.encode(sents)
        for i, qn_cand in enumerate(question_candidates):
            sent_encoding = encodings[i, :, :]
            for j, gap in enumerate(qn_cand.gap_candidates):
                start, end = gap.start_index, gap.end_index
                sliced_embedding = numpy.sum(
                    sent_encoding[start:end, :], axis=0
                )
                sliced_embedding /= math.sqrt(end - start + 1)
                qn_cand.gap_candidates[j].embedding[:] = list(sliced_embedding)
        return question_candidates

    i = 0
    while i < len(spacy_docs):
        questions_candidates = list(
            map(filter_gaps_single_doc, spacy_docs[i: i + batch_size])
        )
        if not elmo_client:
            yield questions_candidates
        questions_candidates = _encode_gaps(questions_candidates, elmo_client)
        for j, qc in enumerate(questions_candidates):
            qc.question_sentence = spacy_docs[j + i].text
        yield questions_candidates
        i += batch_size


def filter_gaps_single_doc(spacy_doc):
    """
    Generates a list of candidates from noun chunks, entities and unigrams.

    Shouldn't be a stopword or duplicate.

    Args:
        spacy_doc: Spacy doc object.
        encode_gaps: whether to encode the gap texts.

    Returns:
        a QuestionCandidate proto containing question sentence and list of
        candidate gaps.
    """
    sent_text = " ".join(tok.text for tok in spacy_doc)
    duplicate_idx = [0] * len(spacy_doc)  # 1 if chosen, 0 otherwise
    candidates = []

    # Consider entities and noun_chunks first.
    # Entities have highest precedence.
    phrases = list(spacy_doc.ents) + list(spacy_doc.noun_chunks)
    phrases = [backoff_phrase(phrase) for phrase in phrases]
    logging.info(
        "Initial Phrase List: %s", "||".join([phr.text for phr in phrases])
    )

    for phrase in phrases:
        if phrase is None or len(phrase) == 0:
            continue

        if sent_text.count(phrase.text) > 2:
            logging.debug(
                "Deleted phrase because it occurs > twice in sentence: ",
                phrase.text,
            )
            continue

        if 1 not in duplicate_idx[phrase.start: phrase.end]:
            candidates.append(phrase)
            duplicate_idx[phrase.start: phrase.end] = [1] * len(phrase)
        else:
            logging.debug("Deleted phrase because duplicate: ", phrase.text)

    # next look at tokens one by one
    for tok in spacy_doc:
        # no stopwords, duplicates and words without vectors are allowed
        if duplicate_idx[tok.i] == 1 or tok.is_stop or tok.is_punct:
            logging.debug(
                "Deleted token because duplicate/stopword/punctuation: ",
                tok.text,
            )
            continue
        if spacy_doc.text.count(tok.text) > 2:
            logging.debug(
                "Deleted token because it occurs > twice"
                " in question sentence: ",
                tok.text,
            )

        # hack to convert spacy token to span.
        candidates.append(spacy_doc[tok.i: tok.i + 1])

    logging.info(
        "Chosen Gaps: %s", "||".join([gap.text for gap in candidates])
    )

    gap_candidates = []
    # look through cands and update gap list.
    for phrase in candidates:
        tags = [word.pos_ for word in phrase]
        gap = question_candidate_pb2.Gap(
            text=phrase.text, start_index=phrase.start,
            end_index=phrase.end, pos_tags=tags
        )
        gap_candidates.append(gap)
    question_candidate = question_candidate_pb2.QuestionCandidate(
        question_sentence=sent_text, gap_candidates=gap_candidates
    )
    return question_candidate


def backoff_phrase(phrase):
    """
    Given a spacy span-phrase, clean it up by removing determiners in front.
    """
    while len(phrase) > 0 and (phrase[0].is_stop and phrase[0].tag_[0] != "N"):
        # sometimes determiners end up in the front
        phrase = phrase[1:]
    return phrase

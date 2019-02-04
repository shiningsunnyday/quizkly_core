""" Filters previous sentences to resolve coreferences. """

AMBIGUOUS_DETERMINERS = [u'this', u'that', u'these', u'such', u'its']


def dep_context(prev_docs, current_doc, question_candidate=None):
    """
    Returns text from prev_docs if spacy doc has unresolved coreferences.

    Args:
        prev_docs: list of spacy docs before `current_doc`.
        current_doc: doc to resolve coreferences form.
        question_candidate: question_candidate to update with context text.
                            optional.
    Returns:
        string with context text. empty is no disambigaution needed/found.
    """
    if not _is_doc_ambiguous(current_doc):
        return ""
    else:
        # return the closest non-ambiguous sentence from prev_sents
        context = ""
        for doc in reversed(prev_docs):
            context = doc.text + " " * (len(context) > 0) + context
            if not _is_doc_ambiguous(doc):
                if question_candidate:
                    question_candidate.context_text = context
                return context
        return ""


def _is_doc_ambiguous(doc):
    """
    Check if a spacy doc is ambiguous
    """
    subj = 0
    # check if sentence starts with a prep or modifier
    # also, make sure it doesnt link to an ambiguous det
    if doc[0].dep_ == 'prep' or (doc[0].dep_[-3:] == 'mod' and doc[0].is_stop):
        subtree = list(doc[0].subtree)
        if len(subtree) == 0:
            return True
        if _ambig_subtree_checker(subtree):
            return True
    for word in doc:
        if word.dep_[0:5] == 'nsubj':
            subj += 1
            # if the subj is a pronoun, and is not preceded by a specific subj.
            if word.pos_ == 'PRON':
                return True
            if _ambig_subtree_checker(word.subtree):
                return True
            # once we come across a concrete subject,
            # we are confident the sentence is unambiguous
            break
        elif word.dep_[0:4] == 'pobj' and subj == 0:
            if _ambig_subtree_checker(word.subtree):
                return True
    return False


def _ambig_subtree_checker(subtree):
    """
    Checks if any of the leafs in a subtree is an anbiguous determiner
    """
    for leaf in subtree:
        if leaf.text.lower() in AMBIGUOUS_DETERMINERS:
            return True
    return False

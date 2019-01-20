"""Script to crawl quizlet api for fill-the-blank quizzes."""

import logging
import pickle
import requests

BLANK_IND = ["__", "BLANK", "blank"]
WORD_DELIM = [",", ";", ";;", "/", ",,", "&", "\n", "-", "and", " "]
QUERIES = [
    "economics_blank",
    "biology_blank",
    "chemistry_blank",
    "fill_the_blank",
    "general_knowledge_blank",
    "history_blank",
    "humanities_blank",
    "physics_blank",
    "politics_blank",
    "law_blank",
]
SEARCH_API_QUERY = (
    "https://api.quizlet.com/2.0/search/sets?client_id=f5kcd8aURS"
    "&q=%s&page=%d&per_page=50"
)
SET_QUERY = "https://api.quizlet.com/2.0/sets/%s?client_id=f5kcd8aURS"


def crawl(q="sociology blank", num_pages=10):
    gap_data = []
    for p in range(num_pages):
        results = requests.get(SEARCH_API_QUERY % (q, p + 1)).json()
        for s in results["sets"]:
            set_id = s["id"]
            candidates = requests.get(SET_QUERY % set_id).json()
            for t in candidates["terms"]:
                if len(t["term"]) > len(t["definition"]):
                    qn = t["term"]
                    gap = t["definition"]
                else:
                    qn = t["definition"]
                    gap = t["term"]
                qn = gap_preprocess(qn)
                tokens = qn.split(" ")
                blank_idx = []
                for i, tok in enumerate(tokens):
                    if any([b in tok for b in BLANK_IND]):
                        blank_idx.append(i)
                if len(blank_idx) < 1:
                    continue
                gap_toks = []
                if len(blank_idx) > 1:
                    for d in WORD_DELIM:
                        if len(gap.split(d)) == len(blank_idx):
                            gap_toks = gap.split(d)
                            gap_toks = [g.strip() for g in gap_toks]
                            for i, idx in enumerate(blank_idx):
                                tokens[idx] = gap_toks[i].strip()
                            # collapse gap toks
                            i = 1
                            while i < len(blank_idx):
                                if blank_idx[i] - blank_idx[i - 1] == 1:
                                    gap_toks[i - 1] += " " + gap_toks[i]
                                    del gap_toks[i]
                                    del blank_idx[i]
                                else:
                                    i += 1
                            break
                else:
                    gap_toks.append(gap)
                    gap_toks = [g.strip() for g in gap_toks]
                    tokens[blank_idx[0]] = gap_toks[0].strip()
                qnSent = " ".join(tokens)

                if len(gap_toks) > 0:
                    gap_data.append([qnSent, qn, gap_toks])
    logging.info("%d samples crawled for %s", len(gap_data), q)
    with open(
        "local/crawled_data/" + q.replace(" ", "_") + "v2.pkl", "wb"
    ) as f:
        pickle.dump(gap_data, f, pickle.HIGHEST_PROTOCOL)


def gap_preprocess(qn):
    add_space_idx = []
    remove_space_idx = []
    for i, char in enumerate(qn):
        if qn[i] == "_" and qn[i - 1] != " " and qn[i - 1] != "_":
            add_space_idx.append(i)
        elif (
            qn[i] == "_"
            and qn[min(i + 1, len(qn) - 1)] != " "
            and qn[min(i + 1, len(qn) - 1)] != "_"
        ):
            add_space_idx.append(i + 1)
        elif qn[i] == " " and qn[i - 1] == "_" and qn[i + 1] == "_":
            remove_space_idx.append(i)
    for i, idx in enumerate(add_space_idx):
        qn = qn[: idx + i] + " " + qn[idx + i:]
    for i, idx in enumerate(remove_space_idx):
        qn = qn[: idx - i] + qn[idx + 1 - i:]
    return qn

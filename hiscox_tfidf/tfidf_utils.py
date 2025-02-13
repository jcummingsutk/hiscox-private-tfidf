from collections import Counter
from dataclasses import dataclass
from functools import partial

from .doc_utils import SpacyModel
from .text_utils import get_unique_words, inverse_document_frequency, tokenize_document


@dataclass
class TFIDF_Info:
    unique_words: list[str]
    corpus: list[str]
    tfidf: list[list[float]]


def term_frequency(text=list[str]) -> dict[str, int]:
    tf = Counter(text)
    return dict(tf)


def compute_idf(
    corpus_as_list_of_list_of_str: list[list[str]], unique_words: list[str]
) -> dict[str, float]:

    return_idf_dict: dict[str, float] = {}
    for word in unique_words:
        try:
            idf_word = inverse_document_frequency(
                word=word,
                corpus_as_lists=corpus_as_list_of_list_of_str,
            )
        except ZeroDivisionError:
            idf_word = (
                -1
            )  # in case some of the words in unique_words are not in the corpus, as will happen in production
        return_idf_dict[word] = idf_word

    return return_idf_dict


def compute_tfidf(
    corpus: list[str],
    nlp: SpacyModel,
) -> TFIDF_Info:

    document_converter = partial(tokenize_document, nlp=nlp)
    corpus_as_list_of_list_of_str = list(map(document_converter, corpus))

    unique_words = get_unique_words(corpus_as_list_of_list_of_str)

    tf_idf_list_of_list = []

    idf = compute_idf(
        corpus_as_list_of_list_of_str=corpus_as_list_of_list_of_str,
        unique_words=unique_words,
    )

    for doc in corpus_as_list_of_list_of_str:
        tf_dict_this_doc = term_frequency(doc)
        tfidf_this_doc = []
        for word in unique_words:
            try:
                tf_word = tf_dict_this_doc[word]
            except KeyError:
                tf_word = 0

            tf_word = float(tf_word * idf[word])
            tfidf_this_doc.append(tf_word)

        tf_idf_list_of_list.append(tfidf_this_doc)

    tfidf_info = TFIDF_Info(
        unique_words=unique_words,
        corpus=corpus_as_list_of_list_of_str,
        tfidf=tf_idf_list_of_list,
    )
    return tfidf_info

from collections import Counter
from dataclasses import dataclass
from math import log

from .doc_utils import SpacyModel, tokenize_corpus
from .text_utils import get_unique_words, num_documents_word_appears


@dataclass
class TFIDF_Info:
    unique_words: list[str]
    corpus: list[str]
    tfidf: list[list[float]]


def term_frequency(text=list[str]) -> dict[str, int]:
    tf = Counter(text)
    return dict(tf)


def inverse_document_frequency_single_word(
    word: str, corpus_as_lists: list[list[str]]
) -> float:
    """For a corpus as a list of list of strings, returns log(N/df(word)), where
    N is the amount of documents in the corpus and df(word) is the number of
    documents where that word appears

    Example: word = "cat", corpus_as_lists=[["dog", "cat"], ["cat", "bird"], ["bird", "dog"]],
    return would be log(3/2)"""
    df_word = float(
        num_documents_word_appears(word=word, corpus_as_lists=corpus_as_lists)
    )
    N = float(len(corpus_as_lists))
    return log(N / df_word)


def compute_idf(
    corpus_as_list_of_list_of_str: list[list[str]], unique_words: list[str]
) -> dict[str, float]:

    return_idf_dict: dict[str, float] = {}
    for word in unique_words:
        try:
            idf_word = inverse_document_frequency_single_word(
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

    corpus_as_list_of_list_of_str = tokenize_corpus(
        corpus=corpus,
        nlp=nlp,
    )

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

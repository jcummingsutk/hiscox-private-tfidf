from math import log

from .doc_utils import SpacyModel, create_document, get_token_list_from_doc
from .token_utils import lemmatize_to_str, remove_punct, remove_stop


def preprocess_text(text: str):
    text = text.lower()
    return text


def tokenize_document(text_document: str, nlp: SpacyModel) -> list[str]:
    """Converts a document to a list of strings with preprocessing, stop word and punctuation removal.

    For example: if document = ["the turtle beat the hare in a race"], the return should be
    ["turtle", "beat", "hare", "race"]"""
    preprocessed = preprocess_text(text=text_document)
    doc = create_document(text=preprocessed, nlp=nlp)
    token_list = get_token_list_from_doc(doc=doc)
    token_list = remove_stop(token_list=token_list)
    token_list = remove_punct(token_list=token_list)
    lemma_list_str = lemmatize_to_str(token_list=token_list)
    return lemma_list_str


def get_unique_words(corpus_as_lists: list[list[str]]) -> list[str]:
    """Gets all of the unique words from a list of list of str.
    Example: corpus_as_lists = [["dog", "cat"], ["cat", "bird"]]
    returns ["dog", "cat", "bird"]"""
    combined_array = []
    for doc_as_list in corpus_as_lists:
        combined_array += doc_as_list  # TODO: Implement without for loop
    unique_words = list(set(combined_array))
    return unique_words


def num_documents_word_appears(word: str, corpus_as_lists: list[list[str]]) -> int:
    """For a corpus as a list of list of strings, returns the number of
    documents a given word is in.

    Example: word = "cat", corpus_as_lists=[["dog", "cat"], ["cat", "bird"]],
    return would be 2"""
    num_appearances = 0
    for doc in corpus_as_lists:
        if word in doc:
            num_appearances += 1
    return num_appearances


def inverse_document_frequency(word: str, corpus_as_lists: list[list[str]]) -> float:
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

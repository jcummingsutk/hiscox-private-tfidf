from functools import partial
from typing import Protocol

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token

from .text_utils import preprocess_text
from .token_utils import lemmatize_to_str, remove_punct, remove_stop


class SpacyModel(Protocol):
    def __call__(text: str) -> Doc:
        """"""


def create_document(text: str, nlp: SpacyModel) -> Doc:
    doc = nlp(text)
    return doc


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


def tokenize_corpus(corpus: list[str], nlp: SpacyModel) -> list[list[str]]:
    """Converts a list of documents"""
    document_converter = partial(tokenize_document, nlp=nlp)
    corpus_as_list_of_list_of_str = list(map(document_converter, corpus))
    return corpus_as_list_of_list_of_str


def get_token_list_from_doc(doc: Doc) -> list[Token]:
    return [token for token in doc]

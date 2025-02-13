from typing import Protocol

from spacy.tokens.doc import Doc
from spacy.tokens.token import Token


class SpacyModel(Protocol):
    def __call__(text: str) -> Doc:
        """"""


def create_document(text: str, nlp: SpacyModel):
    doc = nlp(text)
    return doc


def get_token_list_from_doc(doc: Doc) -> list[Token]:
    return [token for token in doc]

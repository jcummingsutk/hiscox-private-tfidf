from spacy.tokens.token import Token


def remove_stop(token_list: list[Token]) -> list[Token]:
    return [token for token in token_list if not token.is_stop]


def remove_punct(token_list: list[Token]) -> list[Token]:
    return [token for token in token_list if not token.is_punct]


def lemmatize_to_str(token_list: list[Token]) -> list[str]:
    return [token.lemma_ for token in token_list]

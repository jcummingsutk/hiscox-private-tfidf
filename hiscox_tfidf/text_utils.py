def preprocess_text(text: str):
    text = text.lower()
    return text


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

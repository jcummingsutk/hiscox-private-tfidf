from hiscox_tfidf.text_utils import get_unique_words


def test_get_unique_words():
    corpus_as_list_of_lists = [["dog", "cat"], ["cat", "bird"]]
    retrieved_unique_words = get_unique_words(corpus_as_lists=corpus_as_list_of_lists)

    expected_result = ["dog", "cat", "bird"]

    assert len(expected_result) == len(retrieved_unique_words)
    for word in expected_result:
        assert word in retrieved_unique_words

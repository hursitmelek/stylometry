from src.corpus import Chunk
from src.features import build_table, sentence_lengths, tokenize, top_words


def test_tokenize_drops_punctuation_and_case():
    assert tokenize("The cat, the dog!") == ["the", "cat", "the", "dog"]


def test_sentence_lengths_counts_words_per_sentence():
    assert sentence_lengths("One two three. Four five!") == [3, 2]


def test_top_words_is_ordered_by_frequency():
    chunk = Chunk("the the the cat cat dog", "a", "b")
    assert top_words([chunk], n=2) == ["the", "cat"]


def test_punctuation_is_counted_per_thousand_words():
    # 4 words and 1 comma, so the rate is 1 / 4 * 1000
    chunk = Chunk("one, two three four", "a", "b")
    table = build_table([chunk], ["one"])
    assert table["comma"][0] == 250


def test_table_has_one_row_per_chunk():
    chunks = [Chunk("the cat sat.", "a", "b"), Chunk("the dog ran.", "a", "c")]
    table = build_table(chunks, ["the"])
    assert len(table) == 2
    assert list(table["book"]) == ["b", "c"]

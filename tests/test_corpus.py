from pathlib import Path

from src.corpus import clean_text, split_filename, split_into_chunks, strip_gutenberg

BOOK = (
    "The Project Gutenberg eBook of Something\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
    "Call me Ishmael.\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK SOMETHING ***\n"
    "This eBook is for the use of anyone anywhere."
)


def test_strip_gutenberg_keeps_only_the_book():
    text = clean_text(strip_gutenberg(BOOK))
    assert text == "Call me Ishmael."


def test_strip_gutenberg_leaves_plain_text_alone():
    assert strip_gutenberg("Call me Ishmael.") == "Call me Ishmael."


def test_chunks_have_the_right_size():
    text = " ".join(["word"] * 2500)
    chunks = split_into_chunks(text, 1000)
    assert len(chunks) == 2  # the last 500 words are dropped
    assert all(len(c.split()) == 1000 for c in chunks)


def test_filename_gives_author_and_book():
    assert split_filename(Path("doyle_the_lost_world.txt")) == ("doyle", "the_lost_world")

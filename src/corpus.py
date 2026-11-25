from pathlib import Path
import re


class Chunk:
    """A piece of text cut from one book."""

    def __init__(self, text, author, book):
        self.text = text
        self.author = author
        self.book = book

    def __repr__(self):
        return f"Chunk({self.author}, {self.book})"


START_MARK = re.compile(r"\*\*\* ?START OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_MARK = re.compile(r"\*\*\* ?END OF.*?\*\*\*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg(text):
    # Gutenberg files wrap the book in license text, cut it off
    start = START_MARK.search(text)
    if start:
        text = text[start.end():]
    end = END_MARK.search(text)
    if end:
        text = text[:end.start()]
    return text


def clean_text(text):
    # curly quotes would split "don't" into two words
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # only fix spacing, the rest of the punctuation is used as a feature later
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text, size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        piece = words[i:i + size]
        if len(piece) == size:  # drop the short leftover at the end
            chunks.append(" ".join(piece))
    return chunks


def split_filename(path):
    # files are named like doyle_the_sign_of_the_four.txt
    author, _, book = path.stem.partition("_")
    return author, book


def read_book(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    return clean_text(strip_gutenberg(raw))


def load_chunks(folder="data/books", size=1000):
    """Read every book in the folder and cut it into chunks of `size` words."""
    chunks = []
    for path in sorted(Path(folder).glob("*.txt")):
        author, book = split_filename(path)
        text = read_book(path)
        for piece in split_into_chunks(text, size):
            chunks.append(Chunk(piece, author, book))
    return chunks

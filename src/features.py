from collections import Counter
from statistics import mean, stdev
import re

import pandas as pd

# marks we count, a dash can be written in two ways
PUNCTUATION = {
    "comma": ",",
    "semicolon": ";",
    "dash": "-—",
    "exclamation": "!",
    "question": "?",
}

WORD_RE = re.compile(r"[a-z']+")
SENTENCE_END = re.compile(r"[.!?]")


def tokenize(text):
    return WORD_RE.findall(text.lower())


def sentence_lengths(text):
    lengths = []
    for sentence in SENTENCE_END.split(text):
        words = tokenize(sentence)
        if words:
            lengths.append(len(words))
    return lengths


def top_words(chunks, n=100):
    """Find the n most common words in these chunks."""
    counter = Counter()
    for chunk in chunks:
        counter.update(tokenize(chunk.text))
    return [word for word, _ in counter.most_common(n)]


def chunk_features(chunk, words):
    tokens = tokenize(chunk.text)
    total = len(tokens)
    counts = Counter(tokens)

    row = {}
    for word in words:
        row["w_" + word] = counts[word] / total

    lengths = sentence_lengths(chunk.text)
    row["sentence_mean"] = mean(lengths)
    row["sentence_std"] = stdev(lengths) if len(lengths) > 1 else 0.0
    row["word_variety"] = len(counts) / total

    for name, marks in PUNCTUATION.items():
        found = sum(chunk.text.count(m) for m in marks)
        row[name] = found / total * 1000

    return row


def build_table(chunks, words):
    """One row per chunk, one column per feature, plus author and book."""
    rows = []
    for chunk in chunks:
        row = chunk_features(chunk, words)
        row["author"] = chunk.author
        row["book"] = chunk.book
        rows.append(row)
    return pd.DataFrame(rows)

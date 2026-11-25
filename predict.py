"""Guesses the author of a single text."""

import argparse
from pathlib import Path

from src.corpus import Chunk, clean_text, load_chunks, strip_gutenberg
from src.delta import BurrowsDelta
from src.evaluate import feature_columns
from src.features import build_table, top_words

# below this gap the two best authors are too close to call,
# pick the value from the coverage curve that run.py prints
MIN_GAP = 0.05


def train(books="data/books", size=1000, n_words=100):
    chunks = load_chunks(books, size)
    words = top_words(chunks, n_words)
    table = build_table(chunks, words)
    columns = feature_columns(table)
    model = BurrowsDelta().fit(table[columns].values, table["author"].values)
    return model, words, columns


def guess(text, model, words, columns):
    chunk = Chunk(clean_text(strip_gutenberg(text)), "?", "?")
    table = build_table([chunk], words)
    X = table[columns].values

    gap = model.confidence(X)[0]
    ranking = sorted(zip(model.authors, model.distances(X)[0]), key=lambda p: p[1])
    answer = ranking[0][0] if gap >= MIN_GAP else "not sure"
    return answer, gap, ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?")
    parser.add_argument("--file")
    parser.add_argument("--books", default="data/books")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = args.text
    else:
        print("Give a text or use --file")
        return

    words_in_text = len(text.split())
    if words_in_text < 300:
        print(f"Warning: only {words_in_text} words, the guess will be shaky")

    model, words, columns = train(args.books)
    answer, gap, ranking = guess(text, model, words, columns)

    print(f"\nguess: {answer}   (gap {gap:.3f})")
    for author, distance in ranking:
        print(f"  {author:<10} {distance:.3f}")


if __name__ == "__main__":
    main()

"""Runs every experiment and writes the figures into figures/."""

import argparse
from collections import Counter
from pathlib import Path

from src import plots
from src.corpus import load_chunks
from src.delta import BurrowsDelta
from src.evaluate import (MODELS, accuracy, accuracy_by_book, accuracy_interval,
                          coverage_scores, cross_validate, feature_columns)
from src.features import build_table, tokenize, top_words

CHUNK_SIZES = [500, 1000, 2000, 5000]
MAIN_SIZE = 1000


def show_corpus(chunks):
    print("CORPUS")
    books = {(c.author, c.book) for c in chunks}
    per_author = Counter(author for author, _ in books)
    for author in sorted(per_author):
        pieces = sum(1 for c in chunks if c.author == author)
        print(f"  {author:<10} {per_author[author]} books, {pieces} chunks")

    # a real novel uses about 400 different words per 1000, a broken file far fewer
    worst = []
    for author, book in sorted(books):
        first = next(c for c in chunks if c.author == author and c.book == book)
        tokens = tokenize(first.text)
        worst.append((len(set(tokens)) / len(tokens), f"{author}/{book}"))
    worst.sort()
    print("  lowest word variety:", ", ".join(f"{name} {v:.2f}" for v, name in worst[:3]))


def show_results(results, title):
    print(f"\n{title}")
    for model in MODELS:
        low, high = accuracy_interval(results, model)
        print(f"  {model:<9} {accuracy(results, model):.3f}   95% [{low:.3f}, {high:.3f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books", default="data/books")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--words", type=int, default=100)
    args = parser.parse_args()

    figures = Path(args.figures)
    figures.mkdir(exist_ok=True)

    # 1. the corpus
    chunks = load_chunks(args.books, MAIN_SIZE)
    if not chunks:
        print(f"No books found in {args.books}")
        return
    show_corpus(chunks)

    # 2. the main run
    results = cross_validate(chunks, n_words=args.words)
    show_results(results, f"RESULTS AT {MAIN_SIZE} WORDS")

    # 3. which features carry the signal
    print("\nFEATURE GROUPS (logreg)")
    for keep in ["words", "punctuation", "sentences", None]:
        part = cross_validate(chunks, n_words=args.words, keep=keep)
        print(f"  {str(keep):<12} {accuracy(part, 'logreg'):.3f}")

    # 4. chunk size
    print("\nCHUNK SIZE")
    scores = {"delta": [], "logreg": []}
    for size in CHUNK_SIZES:
        part = cross_validate(load_chunks(args.books, size), n_words=args.words)
        for model in scores:
            scores[model].append(accuracy(part, model))
        print(f"  {size:<5} delta {scores['delta'][-1]:.3f}   logreg {scores['logreg'][-1]:.3f}")

    # 5. where it goes wrong
    print("\nWORST BOOKS (delta)")
    print(accuracy_by_book(results, "delta").head(5).to_string())

    # 6. how much better it gets when it may stay silent
    print("\nSTAYING SILENT (delta)")
    points = coverage_scores(results)
    for share, score, gap in points[::5]:
        print(f"  answers {share:.0%} of texts -> {score:.3f} accuracy (gap above {gap:.3f})")

    # 7. figures
    words = top_words(chunks, args.words)
    table = build_table(chunks, words)
    columns = feature_columns(table)
    whole = BurrowsDelta().fit(table[columns].values, table["author"].values)

    plots.confusion_matrix(results, "delta", figures / "confusion_delta.png")
    plots.confusion_matrix(results, "logreg", figures / "confusion_logreg.png")
    plots.chunk_size_curve(CHUNK_SIZES, scores, figures / "chunk_size.png")
    plots.pca_scatter(table, columns, figures / "style_space.png")
    plots.word_heatmap(whole, words, figures / "word_heatmap.png")
    plots.coverage_curve(points, figures / "coverage.png")
    print(f"\nFigures written to {figures}/")


if __name__ == "__main__":
    main()

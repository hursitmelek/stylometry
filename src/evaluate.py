from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.delta import BurrowsDelta
from src.features import PUNCTUATION, build_table, top_words

MODELS = ["delta", "logreg", "majority", "random"]


def make_folds(chunks, n_folds=5, seed=42):
    """Split the books into folds. A book is never in train and test together."""
    books_by_author = {}
    for chunk in chunks:
        books_by_author.setdefault(chunk.author, set()).add(chunk.book)

    rng = np.random.default_rng(seed)
    fold_of_book = {}
    for author, books in books_by_author.items():
        shuffled = rng.permutation(sorted(books))
        # deal the books of one author to the folds one by one,
        # so every fold gets about the same number of books per author
        for i, book in enumerate(shuffled):
            fold_of_book[(author, book)] = i % n_folds

    folds = []
    for f in range(n_folds):
        train = [c for c in chunks if fold_of_book[(c.author, c.book)] != f]
        test = [c for c in chunks if fold_of_book[(c.author, c.book)] == f]
        folds.append((train, test))
    return folds


def feature_columns(table, keep=None):
    """Choose the columns to train on. Used for the ablation experiment."""
    names = [c for c in table.columns if c not in ("author", "book")]
    if keep == "words":
        return [c for c in names if c.startswith("w_")]
    if keep == "punctuation":
        return list(PUNCTUATION)
    if keep == "sentences":
        return ["sentence_mean", "sentence_std", "word_variety"]
    return names


def cross_validate(chunks, n_folds=5, n_words=100, keep=None, seed=42):
    """Train every model on each fold and collect what it predicted."""
    rng = np.random.default_rng(seed)
    parts = []

    for train_chunks, test_chunks in make_folds(chunks, n_folds, seed):
        # the word list comes from the training books only, never from the test books
        words = top_words(train_chunks, n_words)
        train = build_table(train_chunks, words)
        test = build_table(test_chunks, words)

        columns = feature_columns(train, keep)
        X_train, y_train = train[columns].values, train["author"].values
        X_test = test[columns].values
        authors = sorted(set(y_train))

        result = test[["author", "book"]].copy()
        delta = BurrowsDelta().fit(X_train, y_train)
        result["delta"] = delta.predict(X_test)
        result["delta_gap"] = delta.confidence(X_test)

        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        logreg.fit(X_train, y_train)
        result["logreg"] = logreg.predict(X_test)

        # two simple baselines to compare against
        result["majority"] = Counter(y_train).most_common(1)[0][0]
        result["random"] = rng.choice(authors, size=len(test))

        parts.append(result)

    return pd.concat(parts, ignore_index=True)


def accuracy(results, model):
    return (results[model] == results["author"]).mean()


def accuracy_by_book(results, model):
    """Accuracy for every single book, worst first."""
    scores = results.assign(correct=results[model] == results["author"])
    return scores.groupby(["author", "book"])["correct"].mean().sort_values()


def accuracy_interval(results, model, rounds=1000, seed=42):
    """95% interval. We resample books, chunks from one book are not independent."""
    rng = np.random.default_rng(seed)
    correct = results[model] == results["author"]
    by_book = [group.values for _, group in correct.groupby(results["book"])]

    scores = []
    for _ in range(rounds):
        picked = rng.integers(0, len(by_book), len(by_book))
        scores.append(np.concatenate([by_book[i] for i in picked]).mean())
    return np.percentile(scores, [2.5, 97.5])


def coverage_scores(results, steps=20):
    """Accuracy if the model may stay silent on the answers it is least sure about."""
    ordered = results.sort_values("delta_gap", ascending=False)
    correct = (ordered["delta"] == ordered["author"]).values

    points = []
    for k in range(1, steps + 1):
        n = int(len(correct) * k / steps)
        points.append((n / len(correct), correct[:n].mean(), ordered["delta_gap"].values[n - 1]))
    return points

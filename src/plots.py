import matplotlib

matplotlib.use("Agg")  # save files, do not open windows

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def confusion_matrix(results, model, path):
    """Which author gets mistaken for which."""
    authors = sorted(results["author"].unique())
    counts = np.zeros((len(authors), len(authors)))
    for true, guess in zip(results["author"], results[model]):
        counts[authors.index(true)][authors.index(guess)] += 1

    shares = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(authors)):
        for j in range(len(authors)):
            colour = "white" if shares[i][j] > 0.5 else "black"
            ax.text(j, i, f"{shares[i][j]:.2f}", ha="center", va="center", color=colour)

    ax.set_xticks(range(len(authors)), authors, rotation=45, ha="right")
    ax.set_yticks(range(len(authors)), authors)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true author")
    ax.set_title(f"Confusion matrix - {model}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def chunk_size_curve(sizes, scores, path):
    """Accuracy against chunk size, one line per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, values in scores.items():
        ax.plot(sizes, values, marker="o", label=model)

    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("chunk size (words)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("How much text does the model need?")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pca_scatter(table, columns, path):
    """Every chunk as a dot, coloured by author."""
    X = StandardScaler().fit_transform(table[columns].values)
    points = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 5))
    for author in sorted(table["author"].unique()):
        mask = (table["author"] == author).values
        ax.scatter(points[mask, 0], points[mask, 1], s=12, alpha=0.6, label=author)

    ax.set_xlabel("first component")
    ax.set_ylabel("second component")
    ax.set_title("Chunks in style space")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def word_heatmap(model, words, path, top=25):
    """How much each author over- or under-uses the most telling words."""
    authors = model.authors
    profiles = np.array([model.profiles[a] for a in authors])[:, :len(words)]

    # keep the words where the authors disagree the most
    spread = profiles.max(axis=0) - profiles.min(axis=0)
    picked = np.argsort(spread)[-top:][::-1]
    limit = np.abs(profiles[:, picked]).max()

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.imshow(profiles[:, picked], cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(top), [words[i] for i in picked], rotation=45, ha="right")
    ax.set_yticks(range(len(authors)), authors)
    ax.set_title("Word use compared to the average (red = more, blue = less)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def coverage_curve(points, path):
    """Accuracy goes up when the model is allowed to skip its unsure answers."""
    coverage = [p[0] for p in points]
    scores = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(coverage, scores, marker="o")
    ax.set_xlabel("share of texts the model answers")
    ax.set_ylabel("accuracy on those texts")
    ax.grid(alpha=0.3)
    ax.set_title("Letting the model say 'not sure'")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

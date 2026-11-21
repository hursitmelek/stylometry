import numpy as np


class BurrowsDelta:
    """Matches a text to the author whose style profile is closest to it."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # every feature is rescaled with the training mean and spread,
        # otherwise common words like "the" would hide all the others
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds[self.stds == 0] = 1  # a feature that never changes

        z = self.to_z(X)
        self.authors = sorted(set(y.tolist()))  # plain strings, easier to print
        self.profiles = {}
        for author in self.authors:
            self.profiles[author] = z[y == author].mean(axis=0)
        return self

    def to_z(self, X):
        return (np.asarray(X, dtype=float) - self.means) / self.stds

    def distances(self, X):
        """Distance from every text to every author profile."""
        z = self.to_z(X)
        result = np.zeros((len(z), len(self.authors)))
        for i, author in enumerate(self.authors):
            result[:, i] = np.abs(z - self.profiles[author]).mean(axis=1)
        return result

    def predict(self, X):
        closest = self.distances(X).argmin(axis=1)
        return np.array([self.authors[i] for i in closest])

    def confidence(self, X):
        """Gap between the best and the second best author. Small gap means unsure."""
        ordered = np.sort(self.distances(X), axis=1)
        return ordered[:, 1] - ordered[:, 0]

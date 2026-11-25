import numpy as np

from src.delta import BurrowsDelta


def simple_data():
    # two authors that are easy to tell apart
    X = np.array([[1.0, 1.0], [1.2, 0.9], [0.9, 1.1],
                  [5.0, 5.0], [5.2, 4.9], [4.9, 5.1]])
    y = np.array(["austen"] * 3 + ["dickens"] * 3)
    return X, y


def test_learns_both_authors():
    X, y = simple_data()
    model = BurrowsDelta().fit(X, y)
    assert model.authors == ["austen", "dickens"]
    assert (model.predict(X) == y).all()


def test_closest_author_wins():
    X, y = simple_data()
    model = BurrowsDelta().fit(X, y)
    assert model.predict(np.array([[1.0, 1.0]]))[0] == "austen"
    assert model.predict(np.array([[5.0, 5.0]]))[0] == "dickens"


def test_a_feature_that_never_changes_does_not_break_it():
    X, y = simple_data()
    X = np.hstack([X, np.zeros((len(X), 1))])
    model = BurrowsDelta().fit(X, y)
    assert len(model.predict(X)) == len(y)


def test_confidence_is_never_negative():
    X, y = simple_data()
    model = BurrowsDelta().fit(X, y)
    assert (model.confidence(X) >= 0).all()

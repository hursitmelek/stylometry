# Stylometry: who wrote this?

A small authorship attribution system. It reads novels by five writers, cuts them
into equal pieces, turns every piece into numbers that describe writing style, and
then guesses the author of a piece it has never seen.

The point is not to use a ready made library. The feature extraction and the
Burrows' Delta classifier are written from scratch; scikit-learn is only used for
the logistic regression we compare against.

## Data

Five authors, six novels each, all from Project Gutenberg:
Jane Austen, Charles Dickens, Mark Twain, Arthur Conan Doyle, H.G. Wells.

Only novels, no plays, poems or letters. Mixing genres would measure the genre
instead of the writer: a play has very short sentences because it is all dialogue.

Books live in `data/books` and are named `author_title.txt`, for example
`doyle_the_lost_world.txt`. They are committed to the repository, so anyone who
clones it can reproduce the numbers without downloading anything.

## Running it

```
pip install -r requirements.txt
python run.py          # all experiments, writes the figures
python predict.py "some text"
pytest
```

## Features

Each piece of text becomes 108 numbers:

- how often each of the 100 most common words appears
- average sentence length and how much it varies
- how many different words are used
- how often commas, semicolons, dashes, exclamation and question marks appear

Common words carry most of the signal. Writers pick words like *the*, *but* and
*which* without thinking about it, and those habits stay the same whatever the
book is about. Content words would tell us the topic, not the author.

## Models

- **Burrows' Delta** (`src/delta.py`), written from scratch. Every feature is
  turned into a z-score, each author gets an average profile, and a new text is
  given to the nearest profile.
- **Logistic regression** from scikit-learn, as a comparison.
- **Two baselines**: always answer the most common author, and answer at random.
  Without these, an accuracy number means nothing.

## Results

Five authors, 1660 chunks of 1000 words each after balancing. Accuracy is measured
with book-separated cross validation, and the interval comes from resampling books.

| model | accuracy | 95% interval |
| --- | --- | --- |
| logistic regression | 0.920 | 0.890 - 0.947 |
| Burrows' Delta | 0.796 | 0.738 - 0.854 |
| always the biggest author | 0.143 | |
| random guess | 0.188 | |

Chance is 0.20, so both models are well clear of it.

**Which features carry the signal** (logistic regression, 1000 word chunks):

| features used | accuracy |
| --- | --- |
| common words only | 0.877 |
| punctuation only | 0.559 |
| sentence shape only | 0.313 |
| everything | 0.920 |

Common words do most of the work, as expected. The surprise is punctuation: on its
own it is nearly three times better than guessing, so how often a writer reaches
for a semicolon really is a fingerprint.

**Longer texts are easier:**

| chunk size | Delta | logistic regression |
| --- | --- | --- |
| 500 | 0.716 | 0.843 |
| 1000 | 0.796 | 0.920 |
| 2000 | 0.863 | 0.961 |
| 5000 | 0.931 | 0.994 |

**Where it goes wrong.** The worst book is Twain's *The Prince and the Pauper*
(40% correct), followed by Dickens' *A Tale of Two Cities* and Doyle's
*The White Company*. Those are the historical novels in the corpus. When a writer imitates
the speech of another century, the habits we measure change with it, and the model
loses the author.

**Saying nothing is better than guessing.** Delta reports the gap between the best
and the second best author. If it only answers the 30% of texts where that gap is
widest, accuracy goes from 0.80 to 0.99.

## How we avoided fooling ourselves

Three things matter more than the accuracy number.

**Folds are split by book, not by chunk.** Pieces from the same novel are very
similar to each other. If some pieces of a novel were used for training and
others for testing, the model could memorise that novel and the score would look
great while meaning nothing. Every book is either fully in training or fully in
testing.

**The word list is chosen inside each fold.** The 100 most common words are
counted on the training books only. Choosing them on the whole corpus would let
the test books influence which features exist.

**The confidence interval resamples books, not chunks.** Pieces from one book are
not independent observations, so treating them as independent would make the
interval far too narrow.

The model can also stay silent. Burrows' Delta reports the gap between the best
and the second best author; when that gap is small the answer is "not sure".
`run.py` prints how accuracy improves as the model is allowed to skip the
texts it is least sure about.

## Limitations

- Five authors, all writing in English in the 19th and early 20th century.
  Nothing here says it would work on modern text or on other languages.
- The model assumes the real author is one of the five. Given a text by someone
  else it will still name one of them, unless the confidence gap rejects it.
- Sentences are split on `.`, `!` and `?`, so abbreviations like *Mr.* count as
  sentence ends.

## Layout

```
src/corpus.py     reading books, cleaning, cutting into chunks
src/features.py   turning text into numbers
src/delta.py      the Burrows' Delta classifier
src/evaluate.py   folds, cross validation, baselines, ablation, intervals
src/plots.py      the figures
run.py            runs everything
predict.py        guesses the author of one text
```

"""Multinomial Naive Bayes for text classification.

Uses word counts with Laplace smoothing. Suitable for document
classification tasks like spam filtering.
"""
import numpy as np


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter

    def fit(self, documents, labels):
        self.classes = list(set(labels))
        self.vocab = set()
        for doc in documents:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(self.vocab)
        self.vocab_idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)

        self.log_prior = {}
        self.log_likelihood = {}  # {class: array of log P(word|class)}

        for c in self.classes:
            class_docs = [d for d, l in zip(documents, labels) if l == c]
            self.log_prior[c] = np.log(len(class_docs) / len(documents))

            # Count all words in this class
            word_counts = np.zeros(V) + self.alpha  # start with smoothing
            for doc in class_docs:
                for word in doc.lower().split():
                    if word in self.vocab_idx:
                        word_counts[self.vocab_idx[word]] += 1

            # Normalize to log-probabilities
            total = word_counts.sum()
            self.log_likelihood[c] = np.log(word_counts / total)
        return self

    def predict(self, document):
        scores = {}
        words = document.lower().split()
        for c in self.classes:
            score = self.log_prior[c]
            for word in words:
                if word in self.vocab_idx:
                    score += self.log_likelihood[c][self.vocab_idx[word]]
            scores[c] = score
        return max(scores, key=scores.get), scores


if __name__ == "__main__":
    # Build a tiny spam classifier
    spam_emails = [
        "free money click now", "winner selected claim prize",
        "earn cash fast free", "discount offer limited time",
        "free gift card winner", "click here claim reward now",
        "urgent money transfer free", "lottery winner congratulations",
        "free trial offer click", "earn money from home fast",
    ]
    ham_emails = [
        "meeting tomorrow at noon", "project update attached review",
        "code review needed please", "lunch plans for friday team",
        "quarterly report summary data", "bug fix deployed to staging",
        "design review scheduled wednesday", "sprint planning next week",
        "test results look good", "documentation update merged today",
    ]

    docs = spam_emails + ham_emails
    labels = ["spam"]*10 + ["ham"]*10

    clf = MultinomialNB(alpha=1.0).fit(docs, labels)
    prediction, scores = clf.predict("free money winner")
    print(f"Prediction: {prediction}")
    print(f"Log P(spam|doc) = {scores['spam']:.2f}")
    print(f"Log P(ham|doc)  = {scores['ham']:.2f}")

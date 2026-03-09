"""Bernoulli Naive Bayes classifier.

Models binary word presence/absence rather than word counts.
Penalizes the absence of words, making it suitable for short texts.
"""
import numpy as np


class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, documents, labels):
        self.classes = list(set(labels))
        self.vocab = set()
        for doc in documents:
            self.vocab.update(doc.lower().split())
        self.vocab = sorted(self.vocab)
        V = len(self.vocab)

        self.log_prior = {}
        self.log_prob = {}      # log P(word present | class)
        self.log_prob_neg = {}  # log P(word absent | class)

        for c in self.classes:
            class_docs = [d for d, l in zip(documents, labels) if l == c]
            N_c = len(class_docs)
            self.log_prior[c] = np.log(N_c / len(documents))

            # Count documents containing each word
            word_present = np.zeros(V)
            for doc in class_docs:
                words_in_doc = set(doc.lower().split())
                for i, w in enumerate(self.vocab):
                    if w in words_in_doc:
                        word_present[i] += 1

            # Smoothed probabilities
            p = (word_present + self.alpha) / (N_c + 2 * self.alpha)
            self.log_prob[c] = np.log(p)
            self.log_prob_neg[c] = np.log(1 - p)
        return self

    def predict(self, document):
        words_in_doc = set(document.lower().split())
        scores = {}
        for c in self.classes:
            score = self.log_prior[c]
            for i, w in enumerate(self.vocab):
                if w in words_in_doc:
                    score += self.log_prob[c][i]
                else:
                    score += self.log_prob_neg[c][i]  # penalize missing words
            scores[c] = score
        return max(scores, key=scores.get), scores


if __name__ == "__main__":
    # Same spam/ham data as multinomial
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

    bnb = BernoulliNB(alpha=1.0).fit(docs, labels)
    prediction, scores = bnb.predict("free money winner")
    print(f"Prediction: {prediction}")
    print(f"Log P(spam|doc) = {scores['spam']:.2f}")
    print(f"Log P(ham|doc)  = {scores['ham']:.2f}")

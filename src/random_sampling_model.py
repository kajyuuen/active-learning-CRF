import sklearn_crfsuite
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelBinarizer
import copy
import numpy as np
from least_confidence import calculate_least_confidences
import random
import pycrfsuite
from itertools import chain
from sklearn_crfsuite import metrics

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

class RSModel:
    def __init__(self, X_labeled, y_labeled, X_pool, y_pool, query_size = 1):
        random.seed(42)
        self.X_train, self.y_train = X_labeled, y_labeled
        self.X_pool, self.y_pool = X_pool, y_pool
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.set_params({
            'feature.possible_transitions': True
        })
        self.fit()
        self.query_size = query_size

    def fit(self):
        for xseq, yseq in zip(self.X_train, self.y_train):
            self.trainer.append(xseq, yseq)
        self.trainer.train('rs-model.crfsuite')

    def evaluation(self, X_test, y_test):
        tagger = pycrfsuite.Tagger()
        tagger.open('rs-model.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        labels = list(tagger.labels())
        labels.remove('O')

        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

    def query_selection(self):
        if self.query_size > len(self.X_pool):
            print("Empty Pool")
            return 0

        pool_size = len(self.X_pool)
        next_X_pool = copy.deepcopy(self.X_pool)
        next_y_pool = copy.deepcopy(self.y_pool)

        delete_inds = []
        X_train, y_train = [], []
        for random_ind in random.sample(range(pool_size), self.query_size):
            tmp_X = self.X_pool[random_ind]
            tmp_y = self.y_pool[random_ind]
            X_train.append(tmp_X)
            y_train.append(tmp_y)
            delete_inds.append(random_ind)

        delete_inds.sort(reverse = True)
        for delete_ind in delete_inds:
            next_X_pool.pop(delete_ind)
            next_y_pool.pop(delete_ind)

        self.X_pool = next_X_pool
        self.y_pool = next_y_pool
        self.X_train = X_train
        self.y_train = y_train

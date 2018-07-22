import sklearn_crfsuite
from sklearn_crfsuite import metrics
import copy
import numpy as np
from least_confidence import calculate_least_confidences
import random
import pycrfsuite

class RSModel:
    def __init__(self, X_labeled, y_labeled, X_pool, y_pool, query_size = 1):
        random.seed(2)
        self.X_labeled, self.y_labeled = X_labeled, y_labeled
        self.X_pool, self.y_pool = X_pool, y_pool
        self.trainer = pycrfsuite.Trainer(verbose=False)
        self.trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        })
        self.fit()
        self.query_size = query_size

    def fit(self):
        for xseq, yseq in zip(self.X_labeled, self.y_labeled):
            self.trainer.append(xseq, yseq)
        self.trainer.train('rs-model.crfsuite')

    def evaluation(self, X_test, y_test):
        tagger = pycrfsuite.Tagger()
        tagger.open('rs-model.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        return metrics.sequence_accuracy_score(y_test, y_pred)

    def query_selection(self):
        if self.query_size > len(self.X_pool):
            print("Empty Pool")
            return 0

        pool_size = len(self.X_pool)
        next_X_pool = copy.deepcopy(self.X_pool)
        next_y_pool = copy.deepcopy(self.y_pool)

        delete_inds = []
        for random_ind in random.sample(range(pool_size), self.query_size):
            tmp_X = self.X_pool[random_ind]
            tmp_y = self.y_pool[random_ind]
            self.X_labeled.append(tmp_X)
            self.y_labeled.append(tmp_y)
            delete_inds.append(random_ind)

        delete_inds.sort(reverse = True)
        for delete_ind in delete_inds:
            next_X_pool.pop(delete_ind)
            next_y_pool.pop(delete_ind)

        self.X_pool = next_X_pool
        self.y_pool = next_y_pool

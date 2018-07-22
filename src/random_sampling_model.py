import sklearn_crfsuite
from sklearn_crfsuite import metrics
import copy
import numpy as np
from least_confidence import calculate_least_confidences
import random

class RSModel:
    def __init__(self, X_labeled, y_labeled, X_pool, y_pool, crf, query_size = 1):
        random.seed(2)
        self.X_labeled, self.y_labeled = X_labeled, y_labeled
        self.X_pool, self.y_pool = X_pool, y_pool
        self.crf = crf
        self.fit()
        self.query_size = query_size
        self.labels = list(self.crf.classes_)
        self.labels.remove('O')

    def fit(self):
        self.crf.fit(self.X_labeled, self.y_labeled)

    def evaluation(self, X_test, y_test):
        y_pred = self.crf.predict(X_test)
        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=self.labels)

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

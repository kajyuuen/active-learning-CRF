import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import copy
import numpy as np
from least_confidence import calculate_least_confidences
import pycrfsuite

class ALModel:
    def __init__(self, X_labeled, y_labeled, X_pool, y_pool, query_size = 1):
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
        self.trainer.train('al-model.crfsuite')

    def evaluation(self, X_test, y_test):
        tagger = pycrfsuite.Tagger()
        tagger.open('al-model.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        return metrics.sequence_accuracy_score(y_test, y_pred)

    def query_selection(self):
        if self.query_size > len(self.X_pool):
            print("Empty Pool")
            return 0

        tagger = pycrfsuite.Tagger()
        tagger.open('al-model.crfsuite')
        predict_tags = [tagger.tag(xseq) for xseq in self.X_pool]
        probabilities = []
        for x_seq, predict_tag in zip(self.X_pool, predict_tags):
            tagger.set(x_seq)
            probabilities.append(tagger.probability(predict_tag))

        arg_sort_ind = np.argsort(probabilities)[::-1]
        next_X_pool = copy.deepcopy(self.X_pool)
        next_y_pool = copy.deepcopy(self.y_pool)

        delete_inds = []
        for least_confidence_ind in arg_sort_ind[:self.query_size]:
            tmp_X = self.X_pool[least_confidence_ind]
            tmp_y = self.y_pool[least_confidence_ind]
            self.X_labeled.append(tmp_X)
            self.y_labeled.append(tmp_y)
            delete_inds.append(least_confidence_ind)

        delete_inds.sort(reverse = True)
        for delete_ind in delete_inds:
            next_X_pool.pop(delete_ind)
            next_y_pool.pop(delete_ind)

        self.X_pool = next_X_pool
        self.y_pool = next_y_pool

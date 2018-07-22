import sklearn_crfsuite
import argparse
import random
from operator import itemgetter
import nltk
from features import sent2labels, sent2features
from active_learning_model import ALModel
from random_sampling_model import RSModel
import import_conll2003

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    labeled_size = 10
    if(args.test):
        pool_size = 100
        test_size = 100
    else:
        pool_size = int(len(list(import_conll2003.iob_sents('./data/eng.train')))/5)
        test_size = int(len(list(import_conll2003.iob_sents('./data/eng.testb')))/5)

    # Create Dataset
    # labeled_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))[0:labeled_size]
    # pool_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))[labeled_size:labeled_size+pool_size]
    # test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))[0:test_size]
    labeled_sents = random.sample(list(import_conll2003.iob_sents('./data/eng.testa')), labeled_size)
    pool_sents = random.sample(list(import_conll2003.iob_sents('./data/eng.train')), pool_size)
    test_sents = random.sample(list(import_conll2003.iob_sents('./data/eng.testb')), test_size)

    # Convert sentence to features
    X_labeled = [sent2features(s) for s in labeled_sents]
    y_labeled = [sent2labels(s) for s in labeled_sents]
    X_pool = [sent2features(s) for s in pool_sents]
    y_pool = [sent2labels(s) for s in pool_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # Training
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    al_model = ALModel(X_labeled, y_labeled, X_pool, y_pool, crf, pool_sents, 5)
    rs_model = RSModel(X_labeled, y_labeled, X_pool, y_pool, crf, 5)

    al_score = []
    rs_score = []
    print("al_model", al_model.evaluation(X_test, y_test))
    print("rs_model", rs_model.evaluation(X_test, y_test))
    print("--------------------------------------------")
    al_score.append(al_model.evaluation(X_test, y_test))
    rs_score.append(rs_model.evaluation(X_test, y_test))
    for _ in range(300):
        al_model.query_selection()
        al_model.fit()
        print("al_model", al_model.evaluation(X_test, y_test))
        al_score.append(al_model.evaluation(X_test, y_test))
        rs_model.query_selection()
        rs_model.fit()
        print("rs_model", rs_model.evaluation(X_test, y_test))
        rs_score.append(rs_model.evaluation(X_test, y_test))
        print("--------------------------------------------")

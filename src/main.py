import argparse
from operator import itemgetter
import nltk
from features import sent2labels, sent2features
from active_learning_model import ALModel
from random_sampling_model import RSModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    labeled_size = 10
    if(args.test):
        pool_size = 100
        test_size = 100
    else:
        pool_size = -1 * labeled_size - 1
        test_size = -1

    # Create Dataset
    labeled_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))[0:labeled_size]
    pool_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))[labeled_size:labeled_size+pool_size]
    test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))[0:test_size]

    # Convert sentence to features
    X_labeled = [sent2features(s) for s in labeled_sents]
    y_labeled = [sent2labels(s) for s in labeled_sents]
    X_pool = [sent2features(s) for s in pool_sents]
    y_pool = [sent2labels(s) for s in pool_sents]
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    # Training
    al_model = ALModel(X_labeled, y_labeled, X_pool, y_pool)
    rs_model = RSModel(X_labeled, y_labeled, X_pool, y_pool)

    print(al_model.evaluation(X_test, y_test))
    print(rs_model.evaluation(X_test, y_test))
    print("---------------------------------")
    for _ in range(300):
        al_model.query_selection()
        al_model.fit()
        print(al_model.evaluation(X_test, y_test))
        rs_model.query_selection()
        rs_model.fit()
        print(rs_model.evaluation(X_test, y_test))
        print("---------------------------------")
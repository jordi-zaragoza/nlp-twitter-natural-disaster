import numpy as np
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from src.libs.submission import *
from src.libs.preprocess import *
from src.libs.visual import *


def get_model_stats(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False, model_name='ridge'):
    train_vectors, test_vectors = preprocess(train_df, test_df, mix_texts, clean_texts, vectorization, use_LSA)

    clf = model(train_vectors, train_df["target"], model_name)

    mean, std = model_stats(clf, train_vectors, train_df["target"])

    return mean, std


def use_model(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False, model_name='ridge'):
    train_vectors, test_vectors = preprocess(train_df, test_df, mix_texts, clean_texts, vectorization, use_LSA)

    clf = model(train_vectors, train_df["target"], model_name)

    sample_submission = submission(clf, test_vectors)

    visual_check(test_df.text, sample_submission["target"].astype(bool))


def model(X, y, model_name):
    clf = model_select(model_name)
    clf.fit(X, y)
    return clf


def model_select(model_name):
    return {
        'ridge': lambda: linear_model.RidgeClassifier(),
        'logistic': lambda: linear_model.LogisticRegression(penalty='l2'),
        'random_forest': lambda: RandomForestClassifier(),
        'gradient_boosting': lambda: GradientBoostingClassifier(),
    }.get(model_name, lambda: 'Not a valid model name')()


def model_stats(clf, X, y):
    scores = model_selection.cross_val_score(clf, X, y, scoring="f1", cv=5)
    return np.mean(scores), np.std(scores)
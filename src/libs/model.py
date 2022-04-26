from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from libs.submission import *
from libs.preprocess import *
from libs.visual import *


def get_model_stats(train_df, test_df, model_params):
    clf, train_vectors, test_vectors = get_model(train_df, test_df, model_params)
    mean, std = model_stats(clf, train_vectors, train_df["target"])
    return mean, std


def use_model(train_df, test_df, model_params):
    clf, train_vectors, test_vectors = get_model(train_df, test_df, model_params)
    sample_submission = submission(clf, test_vectors)
    visual_check(test_df.text, sample_submission["target"].astype(bool))


def get_model(train_df, test_df, model_params):
    train_vectors, test_vectors = preprocess(train_df,
                                             test_df,
                                             model_params['mix_texts'],
                                             model_params['clean_texts'],
                                             model_params['vectorization'],
                                             model_params['use_LSA'])

    clf = model(train_vectors, train_df["target"], model_params['model_name'])

    return clf, train_vectors, test_vectors


def model(X, y, model_name):
    clf = model_select(model_name)
    print("Training model: ", model_name)
    clf.fit(X, y)
    return clf


def model_select(model_name):
    return {
        'ridge': lambda: linear_model.RidgeClassifier(),
        'logistic': lambda: linear_model.LogisticRegression(penalty='l2', max_iter=500),
        'random_forest': lambda: RandomForestClassifier(),
        'gradient_boosting': lambda: GradientBoostingClassifier(),
        'NN': lambda: MLPClassifier(),
        'svc': lambda: SVC(),
        'Knn': lambda: KNeighborsClassifier(),
        'ada_boost': lambda: AdaBoostClassifier(),
    }.get(model_name, lambda: 'Not a valid model name')()


def model_stats(clf, X, y):
    scores = model_selection.cross_val_score(clf, X, y, scoring="f1", cv=3)
    return scores.mean(), scores.std()
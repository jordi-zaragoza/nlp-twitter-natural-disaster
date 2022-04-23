import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def submission(clf, X_test):
    sample_submission = pd.read_csv("../../data/sample_submission.csv")
    sample_submission["target"] = clf.predict(X_test)
    print(sample_submission.head())
    sample_submission.to_csv("data/submission.csv", index=False)
    return sample_submission
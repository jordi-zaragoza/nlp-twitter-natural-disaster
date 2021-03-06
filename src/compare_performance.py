from libs.best_performance import *
import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

params = {
    # ["ridge", "logistic", "random_forest", 'gradient_boosting', 'NN', 'svc', 'Knn', 'ada_boost']
    'model_name': ["ridge", "logistic", "random_forest", 'gradient_boosting', 'svc', 'Knn', 'ada_boost'],
    'mix_texts': [True, False],     # [True, False]
    'clean_texts': [True, False],   # [True, False]
    'vectorization': ["simple", "tfidf"],     # ["simple", "tfidf"]
    'use_LSA': [True, False]        # [True, False]
}

print(best_performance(train_df,
                       test_df,
                       params))

# -------------------------- PERFORMANCE ----------------------------------------
#     f1_mean    f1_std model_name  mix_texts  clean_texts vectorization  use_LSA
# 0  0.645029  0.028668   logistic      False        False        simple    False
# 1  0.644921  0.028086   logistic      False         True        simple    False
# 2  0.643443  0.030267      ridge      False        False         tfidf    False
# 3  0.639770  0.031792      ridge      False         True         tfidf    False
# 4  0.637960  0.034368   logistic      False         True         tfidf    False

from libs.best_performance import *
import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

params = {
    # ["ridge", "logistic", "random_forest", 'gradient_boosting', 'NN', 'gaussian_process', 'Knn', 'ada_boost']
    'model_name': ["ridge", "logistic", "random_forest", 'gradient_boosting', 'NN', 'gaussian_process', 'Knn', 'ada_boost'],
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
# 0  0.641719  0.046392   logistic      False         True         tfidf    False
# 1  0.639155  0.048322   logistic      False        False         tfidf    False
# 2  0.631778  0.048595      ridge      False         True         tfidf    False
# 3  0.629418  0.056226      ridge      False        False         tfidf    False
# 4  0.625355  0.057197   logistic      False        False        simple    False
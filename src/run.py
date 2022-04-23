from src.libs.best_performance import *
import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

params = {
    'model_name': ["ridge", "logistic", "random_forest", 'gradient_boosting'],
    'mix_texts': [True, False],     # [True, False]
    'clean_texts': [True, False],   # [True, False]
    'vectorization': ["simple", "tfidf"],     # ["simple", "tfidf"]
    'use_LSA': [True, False]        # [True, False]
}

print(best_performance(train_df,
                       test_df,
                       params))

# -------------------------- MY BEST PERFORMANCE --------------------------------------
# f1_mean   f1_std     model_name    mix_texts     clean_texts  vectorization  use_LSA
# 0.641719  0.046392   logistic      False         True         tfidf          False

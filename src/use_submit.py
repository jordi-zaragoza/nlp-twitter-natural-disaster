from libs.model import *
import pandas as pd

train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

model_params = {
    'mix_texts': False,
    'clean_texts': True,
    'vectorization': 'tfidf',
    'use_LSA': False,
    'model_name': 'logistic'
}

use_model(train_df, test_df, model_params)
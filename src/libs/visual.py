import pandas as pd


def visual_check(X, y):
    df_check = pd.DataFrame({'text': X, 'is_disaster': y.astype(bool)})
    print(df_check[df_check.is_disaster is True].sample(5))
    print(df_check[df_check.is_disaster is False].sample(5))
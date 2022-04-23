from src.libs.model import *


def best_performance(train_df, test_df, params):
    results_df = pd.DataFrame()
    for model_name in params['model_name']:
        for mix_texts in params['mix_texts']:
            for clean_texts in params['clean_texts']:
                for vectorization in params['vectorization']:
                    for use_LSA in params['use_LSA']:
                        mean, std = get_model_stats(train_df,
                                                    test_df,
                                                    mix_texts=mix_texts,
                                                    clean_texts=clean_texts,
                                                    vectorization=vectorization,
                                                    use_LSA=use_LSA,
                                                    model_name=model_name)

                        df = pd.DataFrame({"f1_mean": mean,
                                           "f1_std": std,
                                           "model_name": model_name,
                                           "mix_texts": mix_texts,
                                           "clean_texts": clean_texts,
                                           "vectorization": vectorization,
                                           "use_LSA": use_LSA
                                           }, index=[0])

                        results_df = pd.concat([results_df, df])

    results_order = results_df.sort_values(by=['f1_std']).sort_values(by=['f1_mean'], ascending=False)

    return results_order.reset_index(drop=True).head(5)

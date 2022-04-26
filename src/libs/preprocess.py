import regex as re
import pandas as pd
from sklearn import feature_extraction, decomposition


def preprocess(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False):
    if mix_texts:
        train_df, test_df = mix_texts_in_set(train_df, test_df)

    if clean_texts:
        train_df, test_df = clean_texts_in_set(train_df, test_df)

    if "simple" in vectorization:
        train_vectors, test_vectors = simple_vectorize(train_df["text"], test_df["text"])
    elif "tfidf" in vectorization:
        train_vectors, test_vectors = tfidf_vectorize(train_df["text"], test_df["text"])
    else:
        print("no vectorization selected")
    if use_LSA:
        train_vectors, test_vectors = LSA(train_vectors, test_vectors)

    return train_vectors, test_vectors


# --------------------------- Text process --------------------------------

def mix_texts_in_set(train_df_, test_df_):
    train_df, test_df = train_df_.copy(), test_df_.copy()
    train_df['text'] = train_df.apply(lambda x: mix_all_text(x), axis=1)
    test_df['text'] = test_df.apply(lambda x: mix_all_text(x), axis=1)
    return train_df, test_df


def clean_texts_in_set(train_df_, test_df_):
    train_df, test_df = train_df_.copy(), test_df_.copy()
    train_df.text = cleaning_function(train_df.text)
    test_df.text = cleaning_function(test_df.text)
    return train_df, test_df


def mix_all_text(x):
    if not (pd.isna(x.keyword) or pd.isna(x.location)):
        return str(x.keyword) + ' ' + str(x.location) + ' ' + str(x.text)
    else:
        return x.text


def cleaning_function(a):
    """
    function used for clining a list of strings
    """
    a1 = lower_array(a)
    a2 = remove_punct_array(a1)
    return a2


def lower_array(a):
    return [(str(word)).lower() for word in a]


def remove_punct(s):
    # replace - and / by space
    s0 = s.replace('-', ' ').replace('/', ' ')
    # replace 2+ spaces by 1 space
    t = re.compile(r"\s+")
    s1 = t.sub(" ", s0).strip()
    # remove punctuations
    s2 = re.sub(r'[^A-Za-z ]+', '', s1)
    # remove first space
    s3 = re.sub('^\s', '', s2)
    # remove last space
    s4 = s3.rstrip()
    return s4


def remove_punct_array(a):
    return [remove_punct(str(word)) for word in a]


# --------------------------- Vectorizations --------------------------------

def simple_vectorize(train_col, test_col):
    count_vectorizer = feature_extraction.text.CountVectorizer()
    train_vectors = count_vectorizer.fit_transform(train_col)
    test_vectors = count_vectorizer.transform(test_col)
    return train_vectors, test_vectors


# Tf-idf (Term frequency - Inverse document frequency)
def tfidf_vectorize(train_col, test_col):
    count_vectorizer = feature_extraction.text.TfidfVectorizer()
    train_vectors = count_vectorizer.fit_transform(train_col)
    test_vectors = count_vectorizer.transform(test_col)
    return train_vectors, test_vectors


# LSA (Latent Semantic Analysis)
def LSA(train_vectors, test_vectors):
    svd = decomposition.TruncatedSVD(n_components=10, n_iter=20, random_state=42)
    train_matrix = svd.fit_transform(train_vectors)
    test_matrix = svd.transform(test_vectors)
    return train_matrix, test_matrix

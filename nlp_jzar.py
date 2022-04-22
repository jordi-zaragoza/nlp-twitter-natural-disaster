import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing, decomposition
import regex as re

# --------------------------- Best performance ----------------------------
def best_performance(train_df,test_df):
    results_df = pd.DataFrame()
    index_ = 0
    for mix_texts_ in [True, False]:
        for clean_texts_ in [True, False]:
            for vectorization_ in ["simple", "tfidf"]:
                for use_LSA_ in [True, False]:               
                    mean, std = get_model_stats(train_df, 
                                                test_df, 
                                                mix_texts=mix_texts_, 
                                                clean_texts=clean_texts_, 
                                                vectorization=vectorization_, 
                                                use_LSA=use_LSA_)

                    df = pd.DataFrame({"f1_mean":mean,
                                       "f1_std":std,
                                       "mix_texts":mix_texts_, 
                                       "clean_texts":clean_texts_,     
                                       "vectorization":vectorization_,
                                       "use_LSA":use_LSA_
                                      },index=[0])

                    results_df = pd.concat([results_df,df])

    results_order = results_df.sort_values(by=['f1_std']).sort_values(by=['f1_mean'], ascending=False)
    
    return results_order.reset_index(drop=True).head(5) 


def get_model_stats(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False):
    
    train_vectors, test_vectors = preprocess(train_df, test_df, mix_texts, clean_texts, vectorization, use_LSA)

    clf = model_ridge(train_vectors, train_df["target"])
    
    mean, std = model_stats(clf, train_vectors, train_df["target"])
    
    return mean, std


def use_model(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False):
    
    train_vectors, test_vectors = preprocess(train_df, test_df, mix_texts, clean_texts, vectorization, use_LSA)
    
    clf =  model_ridge(train_vectors, train_df["target"])
       
    sample_submission = submission(clf, test_vectors)
    
    visual_check(test_df.text, sample_submission["target"].astype(bool))

# -------------------------------------------------------------------------

def preprocess(train_df, test_df, mix_texts=True, clean_texts=True, vectorization="simple", use_LSA=False):
    if mix_texts:
        train_df, test_df = mix_texts_in_set(train_df, test_df)
        
    if clean_texts:
        train_df, test_df = clean_texts_in_set(train_df, test_df)
    
    if vectorization == "simple":
        train_vectors, test_vectors = simple_vectorize(train_df["text"], test_df["text"])
    elif vectorization == "tfidf":
        train_vectors, test_vectors = tfidf_vectorize(train_df["text"], test_df["text"])
    
    if use_LSA:
        train_vectors, test_vectors = LSA(train_vectors, test_vectors)
        
    return train_vectors, test_vectors

# --------------------------- Text process --------------------------------

def mix_texts_in_set(train_df_, test_df_):
    train_df, test_df = train_df_.copy(), test_df_.copy()
    train_df['text'] = train_df.apply(lambda x: mix_all_text(x) ,axis=1)
    test_df['text'] = test_df.apply(lambda x: mix_all_text(x) ,axis=1)
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

# --------------------------- Model --------------------------------

def model_ridge(X, y):
    clf = linear_model.RidgeClassifier()
    clf.fit(X, y)
    return clf

def model_stats(clf, X, y):
    scores = model_selection.cross_val_score(clf, X, y, scoring="f1", cv=5)
#     print("f1 scores - Mean:", np.mean(scores), "Sdev:", np.std(scores))
    return np.mean(scores), np.std(scores)    

# -------------------------- Submission -----------------------------

def submission(clf, X_test):
    sample_submission = pd.read_csv("data/sample_submission.csv")
    sample_submission["target"] = clf.predict(X_test)
    display(sample_submission.head())
    sample_submission.to_csv("data/submission.csv", index=False)
    return sample_submission

# -------------------------- Vissual Check -----------------------------

def visual_check(X, y):
    df_check = pd.DataFrame({'text':X , 'is_disaster': y.astype(bool)}) 
    display(df_check[df_check.is_disaster == True].sample(5))
    display(df_check[df_check.is_disaster == False].sample(5))


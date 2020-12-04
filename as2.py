#%%
import time
import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
from collections import Counter
import string
from sklearn import linear_model
import ast
import importlib
import numpy as np
from scipy.spatial import distance
import random
import nltk
import pandas as pd

nltk.download("averaged_perceptron_tagger")

#%%
import sys, os

sys.path.insert(0, os.path.abspath("../cse258_hw/"))


#%%
import as2_analysis_utils as as2_analysis

importlib.reload(as2_analysis)


#%%
data_all = list(as2_analysis.parseData_line("renttherunway_final_data.json"))


#%%
def extract_features(data_all, key):
    """
    return data of particualr key
    """
    return [d[key] for d in data_all]


#%%
rating_all = extract_features(data_all, "rating")


#%%
null_count = [i == "null" for i in rating_all]


#%%
sum(null_count) / len(null_count)


#%%
fit_feedbadk = extract_features(data_all, "fit")


#%%
fit_feedbadk_freq = Counter(fit_feedbadk)


#%%
fit_feedbadk_prct = {key: val / len(data_all) for key, val in fit_feedbadk_freq.items()}


#%%
fit_feedbadk_prct

#%% [markdown]
# # Count word frequency

#%%
# top words in fit reviews
fit_reviews_all = [
    d["review_text"] + " " + d["review_summary"] for d in data_all if d["fit"] == "fit"
]
small_reviews_all = [
    d["review_text"] + " " + d["review_summary"]
    for d in data_all
    if d["fit"] == "small"
]
large_reviews_all = [
    d["review_text"] + " " + d["review_summary"]
    for d in data_all
    if d["fit"] == "large"
]


#%%
def tokenize_data_sets(dataset, n=1):
    """
    Each item of dataset is a str
    """
    return [as2_analysis.tokenize_paragraph(d, n=n) for d in dataset]


#%%
# tokenize dataset
fit_tokenized_all = tokenize_data_sets(fit_reviews_all)
small_tokenized_all = tokenize_data_sets(small_reviews_all)
large_tokenized_all = tokenize_data_sets(large_reviews_all)


#%%
fit_word_count = as2_analysis.count_word_freq(fit_tokenized_all)
small_word_count = as2_analysis.count_word_freq(small_tokenized_all)
large_word_count = as2_analysis.count_word_freq(large_tokenized_all)

#%% [markdown]
# # Get most frequent adjective

#%%
def top_n_adj(word_count_dict):
    ## get the most common ajectives
    adj_count = {
        key: word_count_dict[key]
        for key in word_count_dict
        if nltk.pos_tag([key])[0][1][0] == "J"
    }

    # sort them into pairs
    adj_count_pair = [(count, key) for key, count in adj_count.items()]

    # sort
    adj_count_pair.sort(reverse=True)

    return adj_count_pair


#%%
start = time.time()
top_small_adj = top_n_adj(small_word_count)


top_big_adj = top_n_adj(large_word_count)


## get the most common ajectives
adj_fit_count = {
    key: fit_word_count[key]
    for key in fit_word_count
    if nltk.pos_tag([key])[0][1][0] == "J"
}

end = time.time()
print(f"time consume: {end-start}s")
#%%
adj_fit_count_pair = [(count, key) for key, count in adj_fit_count.items()]


#%%
adj_fit_count_pair.sort(reverse=True)


#%%
adj_fit_count_pair[:10]


#%%
top_small_adj[:10]


#%%
top_big_adj[:10]


#%% [markdown]
# Try bigram and trigram

#%%
def top_n_adj_n_gram(word_count_dict):

    ## get the most common ajectives
    adj_count = {
        key: word_count_dict[key]
        for key in word_count_dict
        # if either of the word is an adj, preserve
        if any(
            wordType[0] == "J"
            for _, wordType in nltk.pos_tag(key if type(key) == tuple else (key,))
        )
    }

    # sort them into pairs
    adj_count_pair = list(adj_count.items())

    # # sort
    adj_count_pair.sort(reverse=True, key=lambda x: x[1])

    return adj_count_pair


#%%
def top_adj_pipeline(fit_reviews_all, small_reviews_all, large_reviews_all, n=2):

    fit_tokenized_all = tokenize_data_sets(fit_reviews_all, n)
    small_tokenized_all = tokenize_data_sets(small_reviews_all, n)
    large_tokenized_all = tokenize_data_sets(large_reviews_all, n)

    fit_word_count = as2_analysis.count_word_freq(fit_tokenized_all)
    small_word_count = as2_analysis.count_word_freq(small_tokenized_all)
    large_word_count = as2_analysis.count_word_freq(large_tokenized_all)

    start = time.time()
    top_fit_grams = top_n_adj_n_gram(fit_word_count)
    top_small_grams = top_n_adj_n_gram(small_word_count)
    top_large_grams = top_n_adj_n_gram(large_word_count)
    end = time.time()
    print(f"time consume: {end-start}s")
    return top_fit_grams, top_small_grams, top_large_grams


#%%
top_fit_grams, top_small_grams, top_large_grams = top_adj_pipeline(
    fit_reviews_all, small_reviews_all, large_reviews_all, 2
)
#%%

top_fit_grams[:10]
#%%
top_small_grams[:10]
#%%
top_large_grams[:10]


#%% [markdown]
# # Data Prediction
#%%
random.shuffle(data_all)
#%%
data_size = len(data_all)
valid_percent = 0.2
test_percent = 0.2

#%%
data_train = data_all[: int(data_size * (1 - valid_percent - test_percent))]
data_valid = data_all[
    int(data_size * (1 - valid_percent - test_percent)) : int(
        data_size * (1 - test_percent)
    )
]
data_test = data_all[int(data_size * (1 - test_percent)) :]

#%%
def extract_review(data):
    fit_reviews = [
        d["review_text"] + " " + d["review_summary"] for d in data if d["fit"] == "fit"
    ]
    small_reviews = [
        d["review_text"] + " " + d["review_summary"]
        for d in data
        if d["fit"] == "small"
    ]
    large_reviews = [
        d["review_text"] + " " + d["review_summary"]
        for d in data
        if d["fit"] == "large"
    ]
    return fit_reviews, small_reviews, large_reviews


#%% [markdown]
# Try n-gram BoW logistic regression
#%%
n = 1
fit_reviews_train, small_reviews_train, large_reviews_train = extract_review(data_train)

#%%
top_fit_grams, top_small_grams, top_large_grams = top_adj_pipeline(
    fit_reviews_train, small_reviews_train, large_reviews_train, n
)

#%%
threshold = 500
top_word_set = {word for word, _ in top_fit_grams[:threshold]}.union(
    {word for word, _ in top_small_grams[:threshold]}.union(
        {word for word, _ in top_large_grams[:threshold]}
    )
)
wordId = dict(zip(top_word_set, range(len(top_word_set))))
#%%
def feature(d, n=1):
    feat = [0] * len(top_word_set)
    review = d["review_text"] + " " + d["review_summary"]
    p_list = as2_analysis.tokenize_paragraph(review, n=1)
    for word in p_list:
        if word not in top_word_set:
            continue
        # use BoW for now
        feat[wordId[word]] += 1
    return feat


#%%
def encode_output(data):
    return [0 if d["fit"] == "fit" else 1 if d["fit"] == "small" else 2 for d in data]


#%%
FIT = 0
SMALL = 1
BIG = 2
X_train = [feature(d, n) for d in data_train]
y_train = encode_output(data_train)
X_valid = [feature(d, n) for d in data_valid]
y_valid = encode_output(data_valid)

#%%
model = linear_model.LogisticRegression(C=1000)
#%%
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
#%%
np.mean(y_pred == y_valid)

#%%

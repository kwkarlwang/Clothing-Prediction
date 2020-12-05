#%%
import pickle
import time
import numpy
import urllib
import scipy.optimize
import random
from collections import defaultdict
from collections import Counter
import string
from sklearn import (
    linear_model,
    naive_bayes,
    svm,
    metrics,
    tree,
    neighbors,
    decomposition,
)
import ast
import importlib
import numpy as np
from scipy.spatial import distance
import random
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")


#%%
import sys, os

sys.path.insert(0, os.path.abspath("../cse258_hw/"))


#%%
import as2_analysis_utils as as2_analysis
import as2_plot_utils as as2_plot

importlib.reload(as2_analysis)
importlib.reload(as2_plot)


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
    return [
        as2_analysis.tokenize_paragraph(
            d,
            n=n,
            remove_stopwrods=True,
            stopwords=[
                word
                for word in stopwords.words("english")
                if word != "not" and not ("n'" in word)
            ],
        )
        for d in dataset
    ]


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
def top_n_adj_n_gram(word_count_dict, adj_set=None):

    # if set of adj is given
    if adj_set:
        adj_count = {
            key: word_count_dict[key]
            for key in word_count_dict
            # if either of the word is an adj, preserve
            if key in adj_set
        }
    else:
        ## get the most common ajectives, might be slow
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

    # all adj for fit, small, and large
    fit_adj, small_adj, large_adj = None, None, None

    # load if the pickle exists
    if os.path.exists(f"{n}-grams_adj"):
        with open(f"{n}-grams_adj", "rb") as f:
            fit_adj, small_adj, large_adj = pickle.load(f)

    start = time.time()
    top_fit_grams = top_n_adj_n_gram(fit_word_count, fit_adj)
    top_small_grams = top_n_adj_n_gram(small_word_count, small_adj)
    top_large_grams = top_n_adj_n_gram(large_word_count, large_adj)
    end = time.time()
    print(f"time consume: {end-start}s")
    return top_fit_grams, top_small_grams, top_large_grams


#%%
def write_adj_to_file(n):
    # get all the adj from the entire data
    top_fit_grams, top_small_grams, top_large_grams = top_adj_pipeline(
        fit_reviews_all, small_reviews_all, large_reviews_all, n
    )
    # make into sets
    a, b, c = (
        {word for word, _ in top_fit_grams},
        {word for word, _ in top_small_grams},
        {word for word, _ in top_large_grams},
    )
    # write to file
    with open(f"./{n}-grams_adj", "wb") as f:
        pickle.dump((a, b, c), f)


#%%
top_fit_grams, top_small_grams, top_large_grams = top_adj_pipeline(
    fit_reviews_all, small_reviews_all, large_reviews_all, 3
)
#%%
top_fit_grams[450:500]

#%%
top_small_grams[:20]
#%%
top_large_grams[:20]


#%% [markdown]
# # Feature Engineering


#%%
fit_data = [d for d in data_all if d["fit"] == "fit"]
small_data = [d for d in data_all if d["fit"] == "small"]
large_data = [d for d in data_all if d["fit"] == "large"]
data_balanced = random.sample(fit_data, k=len(large_data)) + small_data + large_data
#%%
random.shuffle(data_balanced)
#%%
data_size = len(data_balanced)
valid_percent = 0.2
test_percent = 0.2


#%%

data_train = data_balanced[: int(data_size * (1 - valid_percent - test_percent))]
data_valid = data_balanced[
    int(data_size * (1 - valid_percent - test_percent)) : int(
        data_size * (1 - test_percent)
    )
]
data_test = data_balanced[int(data_size * (1 - test_percent)) :]

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
# Try n-gram BoW logistic regression/navie bayes


#%%
fit_reviews_train, small_reviews_train, large_reviews_train = extract_review(data_train)

#%%
is_tfidf = True

ns = {1: 100, 2: 500, 3: 1000, 4: 2000}
# ns = {1: 100, 2: 500, 3: 1000, 4: 2000}
top_word_set = {"small", "large", "big"}
idf_score = {}
for n, threshold in ns.items():
    top_fit_grams, top_small_grams, top_large_grams = top_adj_pipeline(
        fit_reviews_train, small_reviews_train, large_reviews_train, n
    )

    a, b, c = (
        {word for word, _ in top_fit_grams[:threshold]},
        {word for word, _ in top_small_grams[:threshold]},
        ({word for word, _ in top_large_grams[:threshold]}),
    )
    # top_gram = (a | b | c) - (a & b & c)
    top_gram = a | b | c
    if is_tfidf:
        idf_score.update(
            as2_analysis.calcualte_idf_score(
                top_gram,
                tokenize_data_sets(
                    fit_reviews_train + small_reviews_train + large_reviews_train, n
                ),
            )
        )
    print(f"unique top {n}-grams: {len(top_gram)}")
    top_word_set |= top_gram

print(f"unique top grams: {len(top_word_set)}")

wordId = dict(zip(top_word_set, range(len(top_word_set))))
#%%
def feature(d, ns=[1], is_tfidf=False):
    feat = [0] * len(top_word_set)
    review = d["review_text"] + " " + d["review_summary"]
    p_list = []
    for n in ns:
        p_list.extend(
            as2_analysis.tokenize_paragraph(
                review,
                n=n,
                remove_stopwrods=True,
                stopwords=[
                    word
                    for word in stopwords.words("english")
                    if word != "not" and not ("n'" in word)
                ],
            )
        )
    for word in p_list:
        if word not in top_word_set:
            continue
        # use BoW for now
        feat[wordId[word]] += 1
    if is_tfidf:
        for word, score in idf_score.items():
            feat[wordId[word]] *= score

    return feat


#%%
def encode_output(data):
    return [labels[d["fit"]] for d in data]


#%%

labels = {"fit": 0, "small": 1, "large": 2}
X_train = [feature(d, ns, is_tfidf) for d in data_train]
y_train = encode_output(data_train)
X_valid = [feature(d, ns, is_tfidf) for d in data_valid]
y_valid = encode_output(data_valid)

#%% [markdown]
# # Modeling
#%% [markdown]
"""
 Try the following numerous text classfication algorithm:
 
    - Navie Bayes
    - SVM
    - Logistic Regression
    - Decision Tree
    - KNN
"""


#%%
models_data = {}
#%% [markdown]
# ## Navie Bayes
#%%
model = naive_bayes.MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
print("Navie Bayes classifcation report\n")
as2_plot.plot_cm(
    metrics.confusion_matrix(y_valid, y_pred, normalize="true"), list(labels.keys())
)
print(metrics.classification_report(y_valid, y_pred))
print()
models_data["naive bayes"] = metrics.classification_report(
    y_valid, y_pred, output_dict=1
)

#%% [markdown]
# Try dimensionality reduction for more time consuimg algorithm
#%%
svd = decomposition.TruncatedSVD(n_components=1000)
X_train_reduce = svd.fit_transform(X_train)
X_valid_reduce = svd.transform(X_valid)

#%% [markdown]
# ## Logistic Regression

#%%

regs = [0.1, 1, 10, 100]
weights = [None]
models = {
    f"C={reg}, weights={weight}": linear_model.LogisticRegression(
        C=reg, class_weight=weight
    )
    for reg in regs
    for weight in weights
}


for desc, model in models.items():
    model.fit(X_train_reduce, y_train)
    y_pred = model.predict(X_valid_reduce)
    print(f"Logistic Regression {desc} classifcation report\n")
    as2_plot.plot_cm(
        metrics.confusion_matrix(y_valid, y_pred, normalize="true"), list(labels.keys())
    )
    print(metrics.classification_report(y_valid, y_pred))
    print()
    models_data[f"logistic regression, {desc}"] = metrics.classification_report(
        y_valid, y_pred, output_dict=1
    )


#%% [markdown]
"""
## SVM

Try the following kernel:

    - linear
    - rbf
    - polynomial
"""
#%%
svd = decomposition.TruncatedSVD(n_components=1000)
X_train_reduce = svd.fit_transform(X_train)
X_valid_reduce = svd.transform(X_valid)


#%%

regs = [0.1, 1, 10]
models = {}
models.update({f"kernel=linear, C={reg}": svm.LinearSVC(C=reg) for reg in regs})
for desc, model in models.items():
    model.fit(X_train_reduce, y_train)
    y_pred = model.predict(X_valid_reduce)
    print(f"SVM {desc} classifcation report and confusion matrix\n")
    as2_plot.plot_cm(
        metrics.confusion_matrix(y_valid, y_pred, normalize="true"), list(labels.keys())
    )
    print(metrics.classification_report(y_valid, y_pred))
    print()
    models_data[f"svm, {desc}"] = metrics.classification_report(
        y_valid, y_pred, output_dict=1
    )

#%%
svd = decomposition.TruncatedSVD(n_components=100)
X_train_reduce = svd.fit_transform(X_train)
X_valid_reduce = svd.transform(X_valid)



#%% [markdown]
# ## Decision Tree

#%%
depths = [5, 10, 20, 50]
models = {
    f"max_depth={depth}": tree.DecisionTreeClassifier(max_depth=depth)
    for depth in depths
}


for desc, model in models.items():
    # delay?
    model.fit(X_train_reduce, y_train)
    y_pred = model.predict(X_valid_reduce)
    print(f"Decision Tree {desc} classifcation report\n")
    as2_plot.plot_cm(
        metrics.confusion_matrix(y_valid, y_pred, normalize="true"), list(labels.keys())
    )
    print(metrics.classification_report(y_valid, y_pred))
    print()
    models_data[f"decision tree, {desc}"] = metrics.classification_report(
        y_valid, y_pred, output_dict=1
    )


#%% [markdown]
# ## K-nearest neighbor
#%%
neighbor_nums = [20, 50, 100, 300]
models = {
    f"n-neighbor={neighbor_num}": neighbors.KNeighborsClassifier(
        n_neighbors=neighbor_num
    )
    for neighbor_num in neighbor_nums
}
for desc, model in models.items():
    model.fit(X_train_reduce, y_train)
    y_pred = model.predict(X_valid_reduce)
    print(f"KNN {desc} classifcation report\n")
    as2_plot.plot_cm(
        metrics.confusion_matrix(y_valid, y_pred, normalize="true"), list(labels.keys())
    )
    print(metrics.classification_report(y_valid, y_pred))
    print()
    models_data[f"knn, {desc}"] = metrics.classification_report(
        y_valid, y_pred, output_dict=1
    )


#%%

"""
Functions to help CSE 258 Assignment 2
"""
import ast
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import gzip
from collections import defaultdict


def parseData_line(path_to_file):
    """function to parse json file line by line, each line 
    of the file is a json structure

    Args:
        path_to_file (str): path to json file

    Yields:
        dict: python dict that contain data from each line
    """
    for line in open(path_to_file):
        line = line.replace("null", '"null"')
        yield ast.literal_eval(line)


def tokenize_paragraph(
    p_str, remove_punc=True, n=1, remove_stopwrods=False, stopwords=None
):
    """Tokenize all words in a paragraph

    Args:
        p_str (str): string of a paragraph
        remove_punc (bool): remove punctuation or not
        n (int): n in n_gram
    Returns:
        list: list of all words in the paragraph
    """
    assert isinstance(remove_punc, bool), "must be bool"
    assert n >= 1, "n must be positive"
    p_str = p_str.lower()

    if remove_punc:
        p_str = [c for c in p_str if not c in string.punctuation]
    else:
        p_str = [c if not c in string.punctuation else " " + c + " " for c in p_str]

    p_str = "".join(p_str)
    p_list = p_str.strip().split()  # already tokenized

    if remove_stopwrods and stopwords is not None:
        p_list = [word for word in p_list if word not in stopwords]

    if n == 1:
        return p_list
    elif n == 2:
        return [bigram for bigram in zip(p_list[:-1], p_list[1:])]
    else:
        return n_gram(p_list, n)


def n_gram(p_list, n=3):
    """Create n-gram given a paragraph

    Args:
        p_list (list): paragraph splitted into list of words
        n: n in n_gram
    Returns:
        list: list of n-gram from p_list
    """
    return [
        tuple(p_list[j] for j in range(i, i + n)) for i in range(len(p_list) - n + 1)
    ]


def count_word_freq(docs_all):
    """Count number of times a word appears

    Args:
        docs_all (list): list of all data tokenized

    Returns:
        dict: ngram_freq, counter how many times a words has appeard
    """
    assert isinstance(docs_all[0][0], str) or isinstance(
        docs_all[0][0], tuple
    ), "input must be tokenized"
    ngram_freq = defaultdict(int)

    # create a frequency dict of all unique ngram
    for doc in docs_all:
        for ngram in doc:
            ngram_freq[ngram] += 1
    return ngram_freq


def compute_confusion_matrix(y_true, y_pred, n_class=None, normalized=False):
    """compute confusion matrix. rows are true labels, cols are predict labels

    Args:
        y_true (array): true labels
        y_pred (array): predict labels
        n_class (int, optional): number of classes. Defaults to None.
        normalized (bool, optional): normalized cm by rows. Defaults to False.
    Returns:
        np array: confusion matrix. 
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    if n_class is None:
        n_class = len(set(y_true))

    cm = np.zeros((n_class, n_class))
    for true, predict in zip(y_true, y_pred):
        cm[int(true), int(predict)] += 1

    if normalized:
        cm /= cm.sum(axis=1).reshape(-1, 1)
    return cm


def calcualte_idf_score(n_gram_set, doc_all):
    """Calculate idf score

    Args:
        n_gram_set (list): list of most common n_grams
        doc_all (list): list of documents, each item is break down into list of 
        n_grams.

    Returns:
        dict: dict for idf score of each n_gram in set
    """
    # assert isinstance(n_gram_set[0],type(doc_all[0][0])) ,"must be same data type"
    idf_score = {}

    doc_all = [set(doc) for doc in doc_all]
    for n_gram in n_gram_set:
        n_doc = 0

        for doc in doc_all:
            if n_gram in doc:
                n_doc += 1

        if n_doc == 0:
            n_doc = 1
        idf_score[n_gram] = np.log10(len(doc_all) / n_doc)

    return idf_score


def train_val_test_pipeline(model, data_all):
    """complete train, validation and test pipeline

    Args:
        model (model type): model for prediction
        data_all (dict): dictionary contains train_x, train_y
                        val_x, val_y
                        test_x,test_y

    Returns:
        tuple: validation_confusion_matrix, test_cm
    """
    assert isinstance(data_all,dict),"data_all must be dictionary"
    # train
    model.fit(data_all["train_x"], data_all["train_y"])

    # predict
    y_val_pred = model.predict(data_all["val_x"])
    y_test_pred = model.predict(data_all["test_x"])

    # compute confusion matrix
    val_cm = compute_confusion_matrix(data_all["val_y"], y_val_pred, normalized=True)
    test_cm = compute_confusion_matrix(data_all["test_y"], y_test_pred, normalized=True)

    return val_cm, test_cm

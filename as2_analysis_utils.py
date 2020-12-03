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


def tokenize_paragraph(p_str, remove_punc=True, n=1):
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
    p_list = p_str.strip().split()

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

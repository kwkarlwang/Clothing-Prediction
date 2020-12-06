"""
Utility functions for plotting
"""
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, class_labels, fmt="g", figsize=(5, 5), title="confusion matrix"):
    """plot confusion matrix using seaborn

    Args:
        cm (np array): confusion matrix
        class_labels (list): list of label names
        fmt (str, optional): format of digit in plot. Defaults to 'g'.
        figsize (tuple, optional): plot figure size. Defaults to (5, 5).
        title (str, optional): title of plot. Defaults to "confusion matrix".
    """
    assert len(cm) == len(class_labels), "incorrect length of class_labels"
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    plt.figure(figsize=figsize)
    sn.heatmap(df_cm, annot=True, cmap="Blues", fmt=fmt, cbar=False)
    plt.ylabel("True Labels", fontsize=14)
    plt.xlabel("Pred Labels", fontsize=14)
    plt.title(title, fontsize=15)
    plt.show()

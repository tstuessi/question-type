"""Utility file for functions that don't belong in a notebook (mainly plotting)
"""
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer

def most_popular_words(data: pd.DataFrame, column: str,
                       grouping_col: str=None, n: int=10,
                       stop_words: List[str]="default"):
    """Generate a plot of the most popular words for a column, either grouped or not grouped

    Args:
        data (pd.DataFrame): pandas dataframe containing column and grouping_col if provided
        column (str): Column containing tokenized values
        grouping_col (str, optional): Column to group by. Defaults to None.
        n (int, optional): Number of results to show. Defaults to 10.
        stop_words (List[str], optional): Words to filter out. Pass None to filter no words. 
            Defaults to ["?", ","]
    """
    if stop_words == "default":
        stop_words = ["?", ","]
    elif stop_words is None:
        stop_words = []

    # first, we need to one-hot encode the list of tokenized words
    mlb = MultiLabelBinarizer()
    transformed_values = mlb.fit_transform(data[column])
    transformed_df = pd.DataFrame(transformed_values, columns=mlb.classes_)

    plt.rc('font', size=15)
    if grouping_col is not None:
        transformed_df[grouping_col] = data[grouping_col]
        return _plot_grouped_popular_words(n, transformed_df, column, grouping_col, stop_words)
    else:
        _plot_single_popular_words(n, transformed_df, column, stop_words)
    plt.show()

def _plot_single_popular_words(n: int, transformed_df: pd.DataFrame, column: str, stop_words: List[str]):
    sums = transformed_df.sum(axis=0)
    sums = sums.drop(stop_words)
    sums_sorted = sums.sort_values(ascending=False)

    plt.figure(figsize=(10,7))
    ax = sums_sorted.head(n).plot(kind="barh")
    ax.invert_yaxis()
    plt.xlabel("Frequency")
    plt.title(f"{n} Most popular words in {column}")

def _plot_grouped_popular_words(n: int, transformed_df: pd.DataFrame, column: str, grouping_col: str, stop_words: List[str]):
    plot_list = []
    for name, group in transformed_df.groupby(grouping_col):
        sums = group.sum(axis=0)
        sums = sums.drop(stop_words + [grouping_col])
        sums_sorted = sums.sort_values(ascending=False)

        plot_list.append((name, sums_sorted.head(n)))
    
    # create a two-column subplot
    # number of rows is calculated from the length of plot list
    ncols = 2
    nrows = len(plot_list) // ncols + len(plot_list) % ncols
    fig = plt.figure(figsize=(ncols*7, nrows*5), constrained_layout=True)
    spec = fig.add_gridspec(nrows, ncols)

    colormap = plt.cm.tab10

    for i, val in enumerate(plot_list):
        current_ax = fig.add_subplot(spec[i // 2, i % 2])
        val[1].plot(kind="barh", ax=current_ax, label=val[0], color=colormap(i))
        current_ax.invert_yaxis()
        current_ax.legend()
    fig.suptitle(f"{n} most popular words in {column}\nby {grouping_col}")

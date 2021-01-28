""" dataset preprocessing

"""
# Author: xxx
# License: xxx

import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
import spacy


def dump(data, out_file):
    """Save data to file

    Parameters
    ----------
    data
    out_file

    Returns
    -------

    """
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)


def load_data(in_file):
    """ load data from file and return the loaded data

    Parameters
    ----------
    in_file

    Returns
    -------
        data: X, y

    """
    with open(in_file, 'rb') as f:
        X, y = pickle.load(f)
    return X, y

def balance(X, y, random_state=42):
    """ sampling data to make balanced classes

    Parameters
    ----------
    X
    y
    random_state

    Returns
    -------
    X_new
    y_new
    """
    class_stats = dict(Counter(y))

    n_thres = min(class_stats.values())
    X_new = []
    y_new = []
    for k, v in class_stats.items():
        replace = True if v < n_thres else False  # oversampling or undersampling
        x = sklearn.utils.resample(X[y == k], replace=replace, n_samples=n_thres, random_state=random_state)
        X_new.extend(x)
        y_new.extend([k]*n_thres)

    return np.asarray(X_new), np.asarray(y_new)

class Dataset():

    def __init__(self, news_text_csv='demo.csv', user_news_clicks_csv=''):
        """parse text files and transform text to numerical data

        Parameters
        ----------
        news_text_csv:
        user_news_clicks_csv:
        """

        self.news_text_csv = news_text_csv
        self.user_news_clicks_csv = user_news_clicks_csv

    def get_data(self, n_rows=100, is_balance=True):
        """parse text files and use spaCy to transform text to vectors

        Parameters
        ----------
        n_rows: the number of lines to be extracted from files
        is_balance: True
            sampling data to get balanced classes
        Returns
        -------
            self
        """
        data_file = os.path.dirname(self.news_text_csv) + '/data.dat'
        if os.path.exists(data_file):
            self.X, self.y = load_data(data_file)

            if is_balance:
                self.X, self.y = balance(self.X, self.y)
            return self

        # 1. parse news_text
        self.news_text_mapping = {}
        with open(self.news_text_csv, 'r') as f:
            line = f.readline()
            i = 0
            while line:
                # if i > n_rows and n_rows !=-1:
                #     break
                if line.startswith('N'):
                    vs = line.split()
                    news_id = vs[0]
                    text = " ".join(vs[1:])
                    self.news_text_mapping[news_id] = text
                else:
                    print(f'skip: {line}')
                line = f.readline()
                i += 1

        # 2. load english tokenizer, tagger, parser, NER and word vectors
        nlp = spacy.load("en_core_web_sm")
        # 3. parse user_news_clicks
        if n_rows == -1: n_rows = None
        df = pd.read_csv(self.user_news_clicks_csv, sep=',', nrows=n_rows)
        X = df[['user_id', 'item']].values
        self.X = []
        for i, v in enumerate(X):
            if i % 1000 == 0:
                print(i, v)
            user_id = int(v[0][1:])
            news_id = v[1]
            if news_id not in self.news_text_mapping.keys():
                print(f"news: {news_id} does not exist!")
                continue
            else:
                n_vector = nlp(self.news_text_mapping[news_id]).vector
                # print(f'i, {n_vector.shape}')
            self.X.append(np.asarray([user_id] + n_vector.tolist()))
        self.X = np.asarray(self.X)
        self.y = df['click'].astype(int).values

        # 4. save the data to file
        dump((self.X, self.y), out_file=data_file)

        # balance data
        if is_balance:
            self.X, self.y = balance(self.X, self.y)

        return self

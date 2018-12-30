#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 17:19
# @Author  : liangxiao
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from example.data_reader import DataParser
from example.data_reader import FeatureDictionary
from example import config


def _load_data():
    df_train = pd.read_csv(config.TRAIN_FILE)
    df_test = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        # record every instance's missing feature number.
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        # to get a new feature by multiply existing features.
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    df_train = preprocess(df_train)
    df_test = preprocess(df_test)

    cols = [c for c in df_train.columns if c not in ["id", "target"]]
    cols = [c for c in cols if c not in config.IGNORE_COLS]

    X_train = df_train[cols].values
    X_test = df_test[cols].values
    y_train = df_train["target"].values
    ids_test = df_test["id"].values
    cat_features_indices = [i for i, c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices


if __name__ == '__main__':
    df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()
    fd = FeatureDictionary(df_train=df_train,
                           df_test=df_test,
                           numeric_cols=config.NUMERIC_COLS,
                           ignored_cols=config.IGNORE_COLS)
    data_parser = DataParser(fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=df_train,has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=df_test)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 17:19
# @Author  : liangxiao
# @Site    : 
# @File    : main.py
# @Software: PyCharm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from example.data_reader import DataParser
from example.data_reader import FeatureDictionary
from example import config
from deep_and_cross import DeepCrossNetwork
from example.log import log


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
    log("start to load data...")
    df_train, df_test, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

    # folds
    log("split folds")
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))
    fd = FeatureDictionary(df_train=df_train,
                           df_test=df_test,
                           numeric_cols=config.NUMERIC_COLS,
                           ignored_cols=config.IGNORE_COLS)
    log("parse data...")
    data_parser = DataParser(fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=df_train, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=df_test)
    field_dim = len(Xi_train[0])
    embedding_dim = 8
    batch_size = 1024
    dcn_params = {
        "field_dim":field_dim,
        "feature_dim":fd.feature_dim,
        "embedding_dim":embedding_dim,
        "dropout_deep":[0.5, 0.5, 0.5],
        "dnn_wides":[32,32],
        "cross_wides":[field_dim * embedding_dim]*2,
        "epoch":30,
        "batch_size":batch_size,
        "learning_rate":0.001,
        "batch_norm":1,
        "batch_norm_decay":0.995,
        "l2_reg":0.01,
        "random_seed":2018,
        "feature_dim":fd.feature_dim,
        "field_dim":len(Xi_train[0]),
        "verbose":True,
        "use_deep":True,
        "use_cross":True
        }

    _get = lambda x, l: [x[i] for i in l]
    y_train_meta = np.zeros((df_train.shape[0],1),dtype=float)
    y_test_meta = np.zeros((df_test.shape[0],1),dtype=float)

    for i,(train_idx, valid_idx) in enumerate(folds):
        # get train/valid sets vial row sampling based on k-folds validation
        log("in fold [ %s ]..."%i)
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        dcn = DeepCrossNetwork(**dcn_params)
        log("began to fit fold %s " % i )
        log("len of y_valid %s" % len(y_valid_))
        dcn.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx, 0] = dcn.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dcn.predict(Xi_test, Xv_test)


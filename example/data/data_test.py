#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/29 17:44
# @Author  : liangxiao
# @Site    : 
# @File    : data_test.py
# @Software: PyCharm
import pandas as pd
if __name__ == '__main__':
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    print df_train
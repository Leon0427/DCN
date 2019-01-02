#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/2 13:35
# @Author  : liangxiao
# @Site    : 
# @File    : log.py
# @Software: PyCharm
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',datefmt='%Y-%m-%d %I:%M:%S')
def log(msg=""):
    logging.debug(msg)

if __name__ == '__main__':
    log("good")


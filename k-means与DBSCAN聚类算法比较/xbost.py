# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     xbost
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/10/22
   Description :  
==================================================
"""
__author__ = 'songdongdong'
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
data = datasets.load_diabetes()
X = dataset[:,0:8]
Y = dataset[:,8]

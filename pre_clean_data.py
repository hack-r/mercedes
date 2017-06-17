from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import random
from sklearn import linear_model
from sklearn.cluster import KMeans
import os
import shutil
import time
import xgboost as xgb
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin


train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')


y = train.y
id0 = train.ID
id1 = test.ID
train.drop(['ID', 'y'], axis=1, inplace=True)
test.drop(['ID', ], axis=1, inplace=True)

print("Dropping constant features...")
for c in train.columns:
    if c == 'ID': continue
    if c == 'y': continue
    if c == 'X0': continue
    if c == 'X1': continue
    if c == 'X2': continue
    if c == 'X3': continue
    if c == 'X4': continue
    if c == 'X5': continue
    if c == 'X6': continue  # there's no X7
    if c == 'X8': continue
    if sum(train[c]) < 20:
        print("Dropping ", c, "because it only had", sum(train[c]), " positive cases out of 4209")
        train.drop(c, axis=1, inplace=True)

print("Dropping redundant features...")
data = train.select_dtypes(['number'])  # dropping non numeric columns
correlationMatrix = data.apply(lambda s: data.corrwith(s))  # finding correlation matrix

highlycorrelated = correlationMatrix > 0.95  # finding highly correlated attributes (cut off
iters = range(len(correlationMatrix.columns) - 1)
drop_cols = []
corr_val = 0.95  # choose the appropriate cut-off value to determine highly correlated features

for i in iters:  # iterate through columns
    for j in range(i):  # iterate through rows
        item = correlationMatrix.iloc[j:(j + 1), (i + 1):(i + 2)]  # finding the cell
        col = item.columns  # storing column number
        row = item.index  # storing row number
        val = item.values  # storing item value
        if val >= corr_val:  # checking if it is highly correlated with the corr_cal alreay declared
            # Prints the correlated feature set and the corr val
            print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
            drop_cols.append(i)  # storing all the column values which are highly correlated

drops = sorted(set(drop_cols))[::-1]  # sort the list of columns to be deleted

for i in drops:
    col = train.iloc[:,
          (i + 1):(i + 2)].columns.values  # Here train is the input df. Hence delete that particular column
    train = train.drop(col, axis=1)

test = test[train.columns]

train['y'] = y
train['ID'] = id0
test['ID'] = id1
train.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train_pre_cleaned.csv', index=False)
test.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test_pre_cleaned.csv', index=False)
# Jason D. Miller

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

random.seed(1337)

print("Model 2: linear_model.ElasticNet")

class feature_eng:
    def doit(self):

        cat_cols = []
        num_cols = []

        train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
        test = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

        y = train.y
        train.drop(['ID', 'y'], axis=1, inplace=True)
        test.drop(['ID', ], axis=1, inplace=True)

        data = pd.concat([train, test])

        for c in data.columns:
            if data[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(data[c].values)
                data[c] = lbl.fit_transform(data[c].values)

                data[c + '_freq'] = data[c].map(data[c].value_counts())
                cat_cols.append(c)
            else:
                num_cols.append(c)

        for c in train.columns:
            if train[c].unique().shape[0] == 1:
                data.drop(c, axis=1, inplace=True)

        train = data[:train.shape[0]]
        test = data[train.shape[0]:]

        return train, y, test

print("Feature engineering...")
fe = feature_eng()
X, y, test = fe.doit()

print("Convert to matrices...")
X = X.as_matrix()
test = test.as_matrix()

print("Set up KFolds...")
n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)
predictions0 = np.zeros((test.shape[0], n_splits))
predictions1 = np.zeros((test.shape[0], n_splits))
score = 0

print("Starting ", n_splits, "-fold CV loop...")
oof_predictions = np.zeros(X.shape[0])
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_valid = X[train_index, :], X[test_index, :]
    y_train, y_valid = y[train_index], y[test_index]

    clf = linear_model.ElasticNet()
    clf.fit(X_train, y_train)

    pred0 = clf.predict(X)
    pred1 = clf.predict(test)
    oof_predictions[test_index] = clf.predict(X_valid)
    predictions0[:, fold] = pred0
    predictions1[:, fold] = pred1
    score += r2_score(clf.predict(X_train), y_train)
    print('Fold %d: Score %f' % (fold, clf.score(X_train, y_train)))

    prediction0 = predictions0.mean(axis=1)
    prediction1 = predictions1.mean(axis=1)
    score /= n_splits
    oof_score = r2_score(y, oof_predictions)

print('=====================')
print('Final Score %f' % score)
print('Final Out-of-Fold Score %f' % oof_score)
print('=====================')

submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y = prediction0
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/insample/model_2_pred_insample.csv', index=False)

submission.y = prediction1
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test/model_2_pred_layer1_test.csv',
                  index=False)

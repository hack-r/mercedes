from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import random
from sklearn.cluster import KMeans
import os
import shutil
import time
import xgboost as xgb
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

random.seed(1337)

print("Model 0: KRR with only base features")

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
                '''
                This is culprit. Increasing CV but decreasing LB score.
                Shouldn't I use value_counts from train data only?
                '''
                cat_cols.append(c)
            else:
                num_cols.append(c)

        for c in train.columns:
            if train[c].unique().shape[0] == 1:
                data.drop(c, axis=1, inplace=True)

        train = data[:train.shape[0]]
        test = data[train.shape[0]:]

        return train, y, test


fe = feature_eng()
X, y, test = fe.doit()

X = X.as_matrix()
test = test.as_matrix()

params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4
params['silent'] = 1
# params['base_score'] = np.mean(y)

for z in np.arange(.1,.6, .2):

    n_splits = 5
    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(X)
    predictions = np.zeros((test.shape[0], n_splits))
    score = 0

    oof_predictions = np.zeros(X.shape[0])
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_valid = X[train_index, :], X[test_index, :]
        y_train, y_valid = y[train_index], y[test_index]

        clf = linear_model.Lasso(alpha=z)
        clf.fit(X_train, y_train)

        pred = clf.predict(test)
        oof_predictions[test_index] = clf.predict(X_valid) #, ntree_limit=model.best_ntree_limit
        predictions[:, fold] = pred
        score += clf.score(X_train, y_train) #.best_score
        print('Fold %d: Score %f'%(fold, clf.score(X_train, y_train))) #    print('Fold %d: Score %f'%(fold, clf)) #


        prediction = predictions.mean(axis=1)
        score /= n_splits
        oof_score = r2_score(y, oof_predictions)

    print('=====================')
    print('x is:', z)
    print('Final Score %f'%score)
    print ('Final Out-of-Fold Score %f'%oof_score)
    print ('=====================')

submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y = prediction
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/mode2_pred.csv', index=False)


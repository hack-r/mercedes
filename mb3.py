from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd
import shutil
import time
import xgboost as xgb

MODEL = 'xgb'


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


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


n_splits = 5
kf = KFold(n_splits=n_splits)
kf.get_n_splits(X)
dtest = xgb.DMatrix(test)
predictions = np.zeros((test.shape[0], n_splits))
score = 0

oof_predictions = np.zeros(X.shape[0])
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_valid = X[train_index, :], X[test_index, :]
    y_train, y_valid = y[train_index], y[test_index]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=False)
    pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
    predictions[:, fold] = pred
    score += model.best_score
    print('Fold %d: Score %f'%(fold, model.best_score))

prediction = predictions.mean(axis=1)
score /= n_splits
oof_score = r2_score(y, oof_predictions)

print('=====================')
print('Final Score %f'%score)
print ('Final Out-of-Fold Score %f'%oof_score)
print ('=====================')

submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y = prediction
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/no_latent.csv',
                    index=False)
"""""
submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y = prediction

output_dir = '../output/%s/%f_%s' % (MODEL, score, time.ctime().replace(' ', '_'))
print
output_dir
os.makedirs(output_dir)
output_file = output_dir + '/%s_%f.csv' % (MODEL, score)
print
output_file
submission.to_csv(output_file, index=False)
cmd = 'cp -v %s %s' % ('./code.py', output_dir + '/code.py')
os.system(cmd)
"""
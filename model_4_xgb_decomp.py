# Jason D. Miller
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# read datasets
print("read datasets...")
train_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

random.seed(1337)

print("Model 0 XGB: Dart XGB with Decomp, 12 comps")

# read datasets
print("read datasets...")
train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test  = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

# process columns, apply LabelEncoder to categorical features
print("process columns, apply LabelEncoder to categorical features...")
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# shape
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

##Add decomposed components: PCA / ICA etc.
print("Add decomposed components: PCA / ICA etc...")

n_pca  = 12
n_ica  = 12
n_nmf  = 12
n_srp  = 12
n_grp  = 12
n_tsvd = 12
n_comp = 12

# NMF
"""""
nmf = NMF(alpha=0.0, beta=1, eta=0.1, init='random', l1_ratio=0.0, max_iter=200,
  n_components=n_nmf, nls_max_iter=2000, random_state=0, shuffle=False,
  solver='cd', sparseness=None, tol=0.0001, verbose=0)
nmf2_results_train = nmf.fit_transform(train.drop(["y"], axis=1).abs())
nmf2_results_test = nmf.transform(test.abs())
"""""

# tSVD
tsvd = TruncatedSVD(n_components=n_tsvd, random_state=42)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)


# PCA
pca = PCA(n_components=n_pca, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_ica, random_state=42,max_iter=1000, tol=.008)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_grp, eps=0.1, random_state=42)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_srp, dense_output=True, random_state=42)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
print("Append PCA components to datasets...")
for i in range(1, n_pca + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

print("Append ICA components to datasets...")
for i in range(1, n_ica + 1):
    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

#print("Append NMF components to datasets...")
#for i in range(1, n_nmf + 1):
#    train['nmf_' + str(i)] = nmf2_results_train[:, i - 1]
#    test['nmf_' + str(i)] = nmf2_results_test[:, i - 1]

print("Append GRP components to datasets...")
for i in range(1, n_grp + 1):
    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

print("Append SRP components to datasets...")
for i in range(1, n_srp + 1):
    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

y_train = train["y"]
y_mean = np.mean(y_train)

# prepare dict of params for xgboost to run with
# https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': .921,  # 0.95, # .921
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1,
    'booster': 'dart',
    'lambda': .3,  # L2 regularization; higher = more conservative
    'alpha': .9  # L1 regularization; higher = more conservative
    # ,'tree_method': 'exact', # Choices: {'auto', 'exact', 'approx', 'hist'}
    # 'grow_policy': 'lossguide' # only works with hist tree_method, Choices: {'depthwise', 'lossguide'}
    # 'normalize_type': 'forest', # DART only; tree or forest, default = "tree"
    # 'rate_drop': 0.1 #[default=0.0] range 0,1
}

print("5 Fold CV and OOF prediction...")
n_splits = 5
kf = KFold(n_splits=n_splits)
X = train.drop(["y"], axis=1)
X = X.drop(["ID"], axis=1)
X = X.as_matrix()
S = test.drop(["ID"], axis=1)
S = S.as_matrix()
y = y_train
# test     = test.as_matrix()
kf.get_n_splits(X)

predictions0 = np.zeros((train.shape[0], n_splits))
predictions1 = np.zeros((test.shape[0], n_splits))
score = 0

oof_predictions = np.zeros(X.shape[0])
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_valid = X[train_index, :], X[test_index, :]
    y_train, y_valid = y[train_index], y[test_index]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_valid, label=y_valid)
    d_test = xgb.DMatrix(S)
    d_train_all = xgb.DMatrix(X, label=y_train)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    print("training model...")
    model = xgb.train(xgb_params, d_train, 1250, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True,
                      verbose_eval=False)
    print("prediction...")
    pred0 = model.predict(d_train_all, ntree_limit=model.best_ntree_limit)
    pred1 = model.predict(d_test, ntree_limit=model.best_ntree_limit)
    oof_predictions[test_index] = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
    predictions0[:, fold] = pred0
    predictions1[:, fold] = pred1
    score += model.best_score
    print('Fold %d: Score %f' % (fold, model.best_score))

prediction0 = predictions0.mean(axis=1)
prediction1 = predictions1.mean(axis=1)
score /= n_splits
oof_score = r2_score(y, oof_predictions)

print('=====================')
print('Final Score %f'%score)
print ('Final Out-of-Fold Score %f'%oof_score)
print ('=====================')

submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')

submission.y = prediction0
submission.columns = ['ID', 'pred_model_4_xgb_decomp']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/insample/model_4_xgb_decomp_pred_insample.csv', index=False)

submission.y = prediction1
submission.columns = ['ID', 'pred_model_4_xgb_decomp']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test/model_4_xgb_decomp_pred_layer1_test.csv', index=False)

"""""

"""""
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVR

train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test  = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

# process columns, apply LabelEncoder to categorical features
print("process columns, apply LabelEncoder to categorical features...")
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

train = train.loc[:,train.apply(pd.Series.nunique) != 1]
test  = test[train.drop(["y"], axis=1).columns]

X = train.drop(["y"], axis=1)
X = X.drop(["ID"], axis=1)
X = X.as_matrix()
S = test.drop(["ID"], axis=1)
S = S.as_matrix()
y = y_train


# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=12)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

svm = SVR(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

#param_grid = dict(features__pca__n_components=[1, 2, 8, 12, 16],
#                  features__univ_select__k=[1, 2,15, 50, 100, 300],
#                  svm__C=[0.1, 1, 10])

param_grid = dict(features__pca__n_components=[15,16,17],
                  features__univ_select__k=[80, 100, 120],
                  svm__C=[1, 2])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_) # 0.526365232922

grid_pred = grid_search.predict(S)

submission = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')

submission.y = grid_search.predict
submission.columns = ['ID', 'pred_model_4_xgb_decomp']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/insample/model_4_xgb_decomp_pred_insample.csv', index=False)

submission.y = grid_pred
submission.columns = ['ID', 'pred_model_4_xgb_decomp']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test/model_4_xgb_decomp_pred_layer1_test.csv', index=False)
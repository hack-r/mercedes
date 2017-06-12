import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD

random.seed(1)

# read datasets
print("read datasets...")
train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

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
#n_nmf  = 7
n_srp  = 8
n_grp  = 1
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
ica = FastICA(n_components=n_ica, random_state=42)
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
    'subsample':  .95,#0.921, # .95
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1,
    'booster': 'dart',
    'lambda': 1,  # L2 regularization; higher = more conservative
    'alpha': 0   # L1 regularization; higher = more conservative
    #,'tree_method': 'exact', # Choices: {'auto', 'exact', 'approx', 'hist'}
    #'grow_policy': 'lossguide' # only works with hist tree_method, Choices: {'depthwise', 'lossguide'}
    #'normalize_type': 'forest', # DART only; tree or forest, default = "tree"
    #'rate_drop': 0.1 #[default=0.0] range 0,1
}

# form DMatrices for Xgboost training
print("form DMatrices for Xgboost training")
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
"""""
print("Cross-validation:")
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   num_boost_round=1500,  # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10,
                   show_stdv=True
                   )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)
"""""
num_boost_rounds = 1300
# train model
print("Train model:")
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# check f2-score (to get higher score - increase num_boost_round in previous cell)
print("Check R2...")
from sklearn.metrics import r2_score

print(r2_score(model.predict(dtrain), dtrain.get_label()))

# make predictions and save results
y_pred = model.predict(dtest)

print("Writing out results...")
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/xgb_12pca_12ica_1grp_8srp_py_500tr_dart_1300br.csv', index=False)
print("Done")

## INSANITY -- neither the CV nor in-sample R2 cor with LB; top kernels make arbitrary choices
# viv696 baseline Kernel Version***                                 .452612812405      LB .56629
# 600 tr, 10 comps, 4md, .95 ss, .005eta, dart booster,             .311173252295 R2 - LB .56474
# 550 tr, 10 comps, 4md, .95 ss, .005eta, default booster,          .304718353255 R2 - LB .56463
# 600 tr, 10 comps, 4md, .95 ss, .005eta, default booster,          .304718353255 R2 - LB .56463
# 500 tr, 14 comps, 4md, .95 ss, .005eta, default booster,          .304718353255 R2 - LB .56463
# 600 tr, 10 PCA/ICA, 7 NMF, 4md, .95 ss, .005eta, dart,            .322492826855 R2 - LB .56420
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, SRP, GRP, 1500br    .494349772241 R2 - LB .56294
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, 8SRP, 1GRP            0.248096080938 R2 - LB .55479
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, 2SRP, 1GRP,         .27xxxx       R2 - LB .55110
# 600 tr, 10 comps, 4md, .95 ss, .005eta, gblinear booster,         .268461797609 R2 - LB .52682
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, SRP, GRP, 1300br    .460313451644 R2 - LB
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, 2SRP, 1GRP, 1300br  .430615485637 R2 - LB
# 500 tr, 12 comps, 4md, .921ss, .005eta, dart, 8SRP, 1GRP, 1300br                R2 - LB

# ***500 tr, 12 comps, 4md, .921ss, .005eta, gbtr, SRP, GRP, PCA, ICA no CV, 1300 num boost rounds

# Worse: gamma, tweedie, L1 > 0, L2 1.1 or .9, hist/loglossguide, 'eta': 0.004, 'eta': 0.006, 11 comps, 10 NMF comps,
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model import LassoLarsCV, BayesianRidge, HuberRegressor, ElasticNetCV, Lasso, LassoCV, LassoLarsIC, \
    LinearRegression, RANSACRegressor, RidgeCV, OrthogonalMatchingPursuitCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prediction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed


# read datasets
print("read datasets...")
train = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test  = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')
samp        = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')


for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

# save columns list before adding the decomposition components

usable_columns = list(set(train.columns) - set(['y']))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

# usable_columns = list(set(train.columns) - set(['y']))

y_train = train['y'].values
y_mean = np.mean(y_train)
id_test = test['ID'].values
# finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

X_train, X_valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.2, random_state=0)

print("form DMatrices for Xgboost training")
dtrain = xgb.DMatrix(X_train.drop('y', axis=1), y_train)
dvalid = xgb.DMatrix(X_valid.drop('y', axis=1), y_valid)
dtest = xgb.DMatrix(test)
dtrainall = xgb.DMatrix(train.drop('y', axis=1), train.y)

'''Train the xgb model then predict the test data'''
random.seed(1)
columns = ['z', 'j', 'result']
result = pd.DataFrame(columns=columns)
try:
    for z in range(450,600,15): #np.arange(.0035, .006, .00025):
        for j in range(1150,1400,50):#np.arange(.92, .96, .002):

            print(z)
            print(j)

            #print("overwrite data sets with raw data")
            #train = train_clean
            #test = test_clean

            xgb_params = {
                'n_trees': 570,
                'eta': 0.0045,
                'max_depth': 4,
                'subsample': 0.93,
                'objective': 'reg:linear',
                'booster': 'dart',
                'eval_metric': 'rmse',
                'base_score': y_mean,  # base prediction = mean(target)
                'silent': 1
            }



            num_boost_rounds = 1200
            # train model
            model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
            y_pred = model.predict(dtest)

            '''Train the stacked models then predict the test data'''

            stacked_pipeline = make_pipeline(
                StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                StackingEstimator(
                    estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55,
                                                        min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
                #StackingEstimator(estimator=BayesianRidge()),
                #StackingEstimator(estimator=ElasticNetCV()),
                #StackingEstimator(estimator=HuberRegressor()),
                #StackingEstimator(estimator=LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False, random_state=None, selection='cyclic')),
                #StackingEstimator(estimator=LassoLarsIC()),
                #StackingEstimator(estimator=LinearRegression()),
                #StackingEstimator(estimator=OrthogonalMatchingPursuitCV()),
                #StackingEstimator(estimator=RANSACRegressor()),
                #StackingEstimator(estimator=RidgeCV()),
                LassoLarsCV()

            )

            stacked_pipeline.fit(finaltrainset, train.y)
            results = stacked_pipeline.predict(finaltestset)

            '''R2 Score on the entire Train data when averaging'''

            print('In-sample R2 score for stacked_pipeline:') # .579, .58
            print(r2_score(train.y, stacked_pipeline.predict(finaltrainset)))
            print('In-sample R2 score for XGB:')
            print(r2_score(y_train, model.predict(dtrain)))
            print('In-sample R2 score for XGB*pipeline:')
            print(r2_score(train.y, stacked_pipeline.predict(finaltrainset)* 0.25 + model.predict(dtrainall) * 0.75))

            '''Average the predition on test data  of both models then save it on a csv file'''

            sub = pd.DataFrame()
            sub['ID'] = id_test
            sub['y'] = y_pred * 0.75 + results * 0.25

            # check f2-score (to get higher score - increase num_boost_round in previous cell)

            print("OOS R2 score for stacked_pipeline:")
            #print(r2_score(y_train, stacked_pipeline.predict(X_train)))
            print("unknown...")

            print("OOS R2 score for XGB:")
            r2 = r2_score(dvalid.get_label(),model.predict(dvalid))
            print(r2)

            print("====================")
            print("z is:", z)
            print("j is:", j)
            print("XGB CV R2 is:", r2)
            print("====================")

            result.loc[len(result)] = [z, j, r2]
            result.to_csv("C:/Users/jmiller/Desktop/loop_results.csv")

            if r2 >= 0.4616:
                print("!!!!!!!!!!!!!!!!!!!!")
                print("FOUND AN IMPROVEMENT")
                print("!!!!!!!!!!!!!!!!!!!!")
                raise StopIteration
except StopIteration:
   pass

sub.to_csv('T://RNA//Baltimore//Jason//ad_hoc//mb//output//jdm_stacked_models.csv', index=False)

# In-sample R2 score for stacked_pipeline:
# 0.605783396205
# In-sample R2 score for XGB:
# 0.704783904699
# In-sample R2 score for XGB*pipeline:
# 0.644203519786
# OOS XGB R2:
# 0.461514
# Public LB: 0.55421

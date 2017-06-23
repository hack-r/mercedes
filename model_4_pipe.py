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
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score

print('model_3_pipe: XGB + pipeline mix with grid searched GRP and NMF component numbers + more stacking estimators')
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
train_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')
samp = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')

for c in train_clean.columns:
    if train_clean[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train_clean[c].values) + list(test_clean[c].values))
        train_clean[c] = lbl.transform(list(train_clean[c].values))
        test_clean[c] = lbl.transform(list(test_clean[c].values))

random.seed(1)
columns = ['z', 'j', 'result']
result = pd.DataFrame(columns=columns)
try:
    for z in range(7, 8, 99):  # np.arange(.0035, .006, .00025): #(0,1,1):#
        for j in range(9, 10, 99):  # np.arange(.92, .96, .002):#(0,1,1):#(
            print(z)
            print(j)

            print("overwrite data sets with raw data")
            train = train_clean
            test = test_clean

            n_pca = 12
            n_ica = 12
            n_nmf = 7 # j selected by grid
            n_srp = 12
            n_grp = 9 # z selected by grid
            n_tsvd = 12
            n_comp = 12

            # NMF
            nmf = NMF(alpha=0.0, beta=1, eta=0.1, init='random', l1_ratio=0.0, max_iter=250,
                      n_components=n_nmf, nls_max_iter=2000, random_state=42, shuffle=False,
                      solver='cd', sparseness=None, tol=0.0001, verbose=1)
            nmf2_results_train = nmf.fit_transform(train.drop(["y"], axis=1).abs())
            nmf2_results_test = nmf.transform(test.abs())

            # tSVD
            tsvd = TruncatedSVD(n_components=n_tsvd, random_state=42)
            tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
            tsvd_results_test = tsvd.transform(test)

            # PCA
            pca = PCA(n_components=n_pca, random_state=42)
            pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
            pca2_results_test = pca.transform(test)

            # ICA
            ica = FastICA(n_components=n_ica, random_state=42, max_iter=1000, tol=.008)
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

            # save columns list before adding the decomposition components
            usable_columns = list(set(train.columns) - set(['y']))

            # Append decomposition components to datasets
            print("Append PCA components to datasets...")
            for i in range(1, n_pca + 1):
                train['pca_' + str(i)] = pca2_results_train[:, i - 1]
                test['pca_' + str(i)] = pca2_results_test[:, i - 1]

            print("Append ICA components to datasets...")
            for i in range(1, n_ica + 1):
                train['ica_' + str(i)] = ica2_results_train[:, i - 1]
                test['ica_' + str(i)] = ica2_results_test[:, i - 1]

            print("Append NMF components to datasets...")
            for i in range(1, n_nmf + 1):
                train['nmf_' + str(i)] = nmf2_results_train[:, i - 1]
                test['nmf_' + str(i)] = nmf2_results_test[:, i - 1]

            print("Append GRP components to datasets...")
            for i in range(1, n_grp + 1):
                train['grp_' + str(i)] = grp_results_train[:, i - 1]
                test['grp_' + str(i)] = grp_results_test[:, i - 1]

            print("Append SRP components to datasets...")
            for i in range(1, n_srp + 1):
                train['srp_' + str(i)] = srp_results_train[:, i - 1]
                test['srp_' + str(i)] = srp_results_test[:, i - 1]

            '''
            train['arules_group1'] = train.X194 + train.X187 + train.X85 + train.X283 + train.X154 + train.X374 + train.X321
            train['arules_group2'] = train.X50  + train.X129 + train.X49 + train.X263 + train.X137 + train.X324 + train.X70  + train.X361 + train.X205 + train.X58 + train.X136 + train.X74
            train['arules_group3'] = train.X161 + train.X202 + train.X45 + train.X377 + train.X356 + train.X186 + train.X362 + train.X334 + train.X133

            test['arules_group1'] = test.X194 + test.X187 + test.X85 + test.X283 + test.X154 + test.X374 + test.X321
            test['arules_group2'] = test.X50  + test.X129 + test.X49 + test.X263 + test.X137 + test.X324 + test.X70  + test.X361 + test.X205 + test.X58 + test.X136 + test.X74
            test['arules_group3'] = test.X161 + test.X202 + test.X45 + test.X377 + test.X356 + test.X186 + test.X362 + test.X334 + test.X133
            usable_columns.append('arules_group1')
            usable_columns.append('arules_group2')
            usable_columns.append('arules_group3')
            '''

            y_train0 = train['y'].values
            y_train = train['y'].values
            y_mean = np.mean(y_train)
            id_test = test['ID'].values
            # finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays)
            finaltrainset = train[usable_columns].values
            finaltestset = test[usable_columns].values

            X_train, X_valid, y_train, y_valid = train_test_split(train, y_train, test_size=0.2, random_state=0)
            X_train1, X_valid1, y_train1, y_valid1 = train_test_split(train, y_train0, test_size=0.2, random_state=10)
            X_train2, X_valid2, y_train2, y_valid2 = train_test_split(train, y_train0, test_size=0.2, random_state=100)
            X_train3, X_valid3, y_train3, y_valid3 = train_test_split(train, y_train0, test_size=0.2, random_state=1000)
            X_train4, X_valid4, y_train4, y_valid4 = train_test_split(train, y_train0, test_size=0.2,
                                                                      random_state=10000)

            print("form DMatrices for Xgboost training")
            dtrain = xgb.DMatrix(X_train.drop('y', axis=1), y_train)
            dvalid = xgb.DMatrix(X_valid.drop('y', axis=1), y_valid)
            dvalid1 = xgb.DMatrix(X_valid1.drop('y', axis=1), y_valid1)
            dvalid2 = xgb.DMatrix(X_valid2.drop('y', axis=1), y_valid2)
            dvalid3 = xgb.DMatrix(X_valid3.drop('y', axis=1), y_valid3)
            dvalid4 = xgb.DMatrix(X_valid4.drop('y', axis=1), y_valid4)
            dtest = xgb.DMatrix(test)
            dtrainall = xgb.DMatrix(train.drop('y', axis=1), train.y)

            '''Train the xgb model then predict the test data'''

            xgb_params = {
                'n_trees': 500,
                'eta': 0.0045,
                'max_depth': 4,
                'subsample': 0.93,
                'objective': 'reg:linear',
                'booster': 'dart',
                'eval_metric': 'rmse',
                'base_score': y_mean,  # base prediction = mean(target)
                'silent': 1
            }

            num_boost_rounds = 1225
            # train model
            model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
            y_pred = model.predict(dtest)

            print('In-sample R2 score for XGB:')
            print(r2_score(y_train, model.predict(dtrain)))

            print("OOS R2 score for XGB:")
            r2 = r2_score(dvalid.get_label(), model.predict(dvalid))
            print(r2)

            '''Train the stacked models then predict the test data'''

            stacked_pipeline = make_pipeline(
                StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3,
                                                                      max_features=0.55,
                                                                      min_samples_leaf=18, min_samples_split=14,
                                                                      subsample=0.7)),
                 StackingEstimator(estimator=BayesianRidge()),
                 StackingEstimator(estimator=ElasticNetCV()),
                 StackingEstimator(estimator=HuberRegressor()),
                 StackingEstimator(estimator=LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=True, precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1, positive=False, random_state=None, selection='cyclic')),
                 StackingEstimator(estimator=LassoLarsIC()),
                 StackingEstimator(estimator=LinearRegression()),
                 StackingEstimator(estimator=OrthogonalMatchingPursuitCV()),
                 StackingEstimator(estimator=RANSACRegressor()),
                 StackingEstimator(estimator=RidgeCV()),
                # LassoLarsCV()      # .6
                # LinearRegression() # .6
                # ElasticNetCV()     # worse
                # OrthogonalMatchingPursuitCV() # worse
                # LassoLarsIC()      # 0.589969080645
                # BayesianRidge()    # 0.584972437762
                # HuberRegressor()   # 0.538743277168
                # BaggingRegressor() # 0.915862724858...
                # AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), learning_rate=.01, n_estimators=300, random_state=1337) #0.641126381615
                # RandomForestRegressor() # 0.902507869334
                # neural_network.MLPRegressor() # 0.525858424976
                StackingEstimator(estimator=AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), learning_rate=.01, n_estimators=300, random_state=1337)),
                StackingEstimator(estimator=BaggingRegressor()),
                StackingEstimator(estimator=RandomForestRegressor()),
                LinearRegression()
            )

            stacked_pipeline.fit(finaltrainset, train.y)
            results = stacked_pipeline.predict(finaltestset)

            '''R2 Score on the entire Train data when averaging'''

            print('In-sample R2 score for stacked_pipeline:')  # .579, .58
            print(r2_score(train.y, stacked_pipeline.predict(finaltrainset)))
            print('In-sample R2 score for XGB:')
            print(r2_score(y_train, model.predict(dtrain)))
            print('In-sample R2 score for XGB*pipeline:')
            print(r2_score(train.y, stacked_pipeline.predict(finaltrainset) * 0.25 + model.predict(dtrainall) * 0.75))

            '''Average the predition on test data  of both models then save it on a csv file'''

            sub = pd.DataFrame()
            sub['ID'] = id_test
            sub['y'] = y_pred * 0.75 + results * 0.25

            print("OOS R2 score for XGB:")
            r20 = r2_score(dvalid.get_label(), model.predict(dvalid))
            r21 = r2_score(dvalid1.get_label(), model.predict(dvalid1))
            r22 = r2_score(dvalid2.get_label(), model.predict(dvalid2))
            r23 = r2_score(dvalid3.get_label(), model.predict(dvalid3))
            r24 = r2_score(dvalid4.get_label(), model.predict(dvalid4))
            r2 = np.mean([r20, r21, r22, r23, r24])
            print(r2)

            print("OOS R2 score for stacked_pipeline:")
            # print(r2_score(y_train, stacked_pipeline.predict(X_train)))
            print("unknown...")

            print("====================")
            print("z is:", z)
            print("j is:", j)
            print("XGB CV R2 is:", r2)
            print("====================")

            result.loc[len(result)] = [z, j, r2]
            result.to_csv("C:/Users/jmiller/Desktop/loop_results.csv")

            if r2 >= 1:  # 0.465932:
                print("!!!!!!!!!!!!!!!!!!!!")
                print("FOUND AN IMPROVEMENT")
                print("!!!!!!!!!!!!!!!!!!!!")
                raise StopIteration
except StopIteration:
    pass
print("Loop complete.")


sub.to_csv('T://RNA//Baltimore//Jason//ad_hoc//mb//output//model_4_pipe.csv', index=False)

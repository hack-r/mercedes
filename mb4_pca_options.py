import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

random.seed(1)
columns = ['z','j', 'result']
result = pd.DataFrame(columns=columns)
for z in range(1):
    for j in range(1,5):
        n_pca = 10
        n_ica = 10

        print(z)
        print(j)

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
        from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
        print("Add decomposed components: PCA / ICA etc...")
        #n_pca_comp = 10
        #n_ica_comp = 10
        #n_nmf_comp = 10

        # PCA
        pca = PCA(n_components=n_pca , random_state=42)
        pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
        pca2_results_test = pca.transform(test)

        # ICA
        ica = FastICA(n_components=n_ica, random_state=42)
        ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
        ica2_results_test = ica.transform(test)

        # Append decomposition components to datasets
        print("Append PCA components to datasets...")
        for i in range(1, n_pca + 1):
            train['pca_' + str(i)] = pca2_results_train[:, i - 1]
            test['pca_' + str(i)] = pca2_results_test[:, i - 1]

        print("Append ICA components to datasets...")
        for i in range(1, n_ica + 1):
            train['ica_' + str(i)] = ica2_results_train[:, i - 1]
            test['ica_' + str(i)] = ica2_results_test[:, i - 1]

        y_train = train["y"]
        y_mean = np.mean(y_train)

        # prepare dict of params for xgboost to run with
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        xgb_params = {
            'n_trees': 600,
            'eta': 0.005,
            'max_depth': 4,
            'subsample': 0.95,
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
        print("Cross-validation:")
        cv_result = xgb.cv(xgb_params,
                           dtrain,
                           num_boost_round=1500,  # increase to have better results (~700)
                           early_stopping_rounds=50,
                           verbose_eval=100,
                           show_stdv=True
                           )

        num_boost_rounds = len(cv_result)
        print(num_boost_rounds)

        # train model
        print("Train model:")
        model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

        # check f2-score (to get higher score - increase num_boost_round in previous cell)
        print("Check R2...") # .311173252295
        from sklearn.metrics import r2_score

        print(r2_score(model.predict(dtrain), dtrain.get_label()))
        r = r2_score(model.predict(dtrain), dtrain.get_label())
        print("placeholder is:",z)
        print("nmf is:",j)
        print("r2 is:",r)
        result.loc[len(result)] = [z, j, r]
        if r > .311173252295:
            break

# make predictions and save results
y_pred = model.predict(dtest)

print("Writing out results...")
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/xgb_10pca_ica_py_600tr_dart.csv', index=False)
print("Done")

# 600 tr, 10 comps, 4md, .95 ss, .005eta, dart booster,     .311173252295 R2 CV - LB .56474
# 550 tr, 10 comps, 4md, .95 ss, .005eta, default booster,  .304718353255 R2 CV - LB .56463
# 600 tr, 10 comps, 4md, .95 ss, .005eta, default booster,  .304718353255 R2 CV - LB .56463
# 500 tr, 14 comps, 4md, .95 ss, .005eta, default booster,  .304718353255 R2 CV - LB .56463
# 600 tr, 10 PCA/ICA, 7 NMF, 4md, .95 ss, .005eta, dart,    .322492826855 R2 CV - LB .56420
# 600 tr, 10 comps, 4md, .95 ss, .005eta, gblinear booster, .268461797609 R2 CV - LB .52682

# Worse: gamma, tweedie, L1 > 0, L2 1.1 or .9, hist/loglossguide, 'eta': 0.004, 'eta': 0.006, 11 comps, 10 NMF comps, 10 TruncatedSVD
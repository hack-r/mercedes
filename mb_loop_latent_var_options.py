import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.model_selection import train_test_split

# read datasets
print("read datasets...")
train_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test_clean = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')

random.seed(1)
columns = ['z', 'j', 'result']
result = pd.DataFrame(columns=columns)
try:
    for z in range(200,500,100):
        for j in np.arange(0.0001,0.001, 0.0001):

            print(z)
            print(j)

            print("overwrite data sets with raw data")
            train = train_clean
            test = test_clean

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
            ica = FastICA(n_components=n_ica, random_state=42,max_iter=z, tol=j)
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

            X_train, X_valid, y_train, y_valid = train_test_split(train, y_train, test_size = 0.1, random_state = 0)

            # prepare dict of params for xgboost to run with
            # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
            xgb_params = {
                'n_trees': 500,
                'eta': 0.005,
                'max_depth': 4,
                'subsample':  .921, #0.95, # .921
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
            dtrain = xgb.DMatrix(X_train.drop('y', axis=1), y_train)
            dvalid = xgb.DMatrix(X_valid.drop('y', axis=1), y_valid)
            dtest  = xgb.DMatrix(test)

            # xgboost, cross-validation
            print("Cross-validation:")
            cv_result = xgb.cv(xgb_params,
                               dtrain,
                               num_boost_round=1500,  # increase to have better results (~700)
                               early_stopping_rounds=50,
                               verbose_eval=50,
                               show_stdv=True
                               )

            #num_boost_rounds = len(cv_result)
            print(num_boost_rounds)
            r = cv_result.min()[0]

            num_boost_rounds = 1300
            # train model
            print("Training model...")
            model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

            # check f2-score (to get higher score - increase num_boost_round in previous cell)
            print("Calculating CV R2...")
            from sklearn.metrics import r2_score

            r2 = r2_score(dvalid.get_label(),model.predict(dvalid))

            print("z is:",z)
            print("j is:",j)
            print("CV rmse is:", r)
            print("CV R2 is:", r2)

            result.loc[len(result)] = [z, j, r]
            if r < 8.4 and r2 > 0.460313451644:
                print("!!!!!!!!!!!!!!!!!!!!")
                print("FOUND AN IMPROVEMENT")
                print("!!!!!!!!!!!!!!!!!!!!")
                #raise StopIteration
except StopIteration:
   pass

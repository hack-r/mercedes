# Jason D. Miller
import numpy as np
from sklearn import neural_network, linear_model
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, \
    IsolationForest, RandomTreesEmbedding, RandomForestRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
import os, glob
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
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

# read datasets
print("read datasets...")
train   = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/train.csv')
test    = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/test.csv')
sample  = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
df      = sample.drop('y',axis=1) # just a place to put the predictions, b/c test / train happen to be the same length
df_test = sample.drop('y',axis=1) # just a place to put the predictions, b/c test / train happen to be the same length

print("Ensemble set up ...")
# Create training data
os.chdir("T:/RNA/Baltimore/Jason/ad_hoc/mb/insample")
files = glob.glob('*.csv')
dfs = {}

# list input files
for f in files:
    dfs[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f)

# build training data set
for c in dfs.keys():
    pred = dfs[c].iloc[:, [1]]
    df   = pd.concat([df, pred], axis=1)

df = df.drop('ID', axis = 1) # this ID doesn't correspond to the training data anyway

# Create test data
del(files)
os.chdir("T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test")
files = glob.glob('*.csv')
dfs_test = {}

# list input files
for f in files:
    dfs_test[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f)

# build test data set
for c in dfs_test.keys():
    pred = dfs_test[c].iloc[:, [1]]
    df_test   = pd.concat([df_test, pred], axis=1)

id      = df_test['ID']
df_test = df_test.drop('ID', axis = 1) # this ID doesn't correspond to the training data anyway

# Make columns have same order
df_test = df_test[df.columns]


# Drop ridiculously bad data
d = df.describe()
for c in d.columns:
    if d[c]['min'] < -100 or d[c]['max'] > 1000:
        df      = df.drop(c, axis=1)
        df_test = df_test.drop(c, axis=1)
df.shape
df_test.shape

print("Ensemble Model 0: AdaBoostRegressor")
ens0 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), learning_rate=.05,
                          n_estimators=300, random_state=1337)

ens0.fit(df, train.y)

# In Sample R2
ens0_insample_pred = ens0.predict(df)
r2_score(train.y, ens0_insample_pred ) # 0.62334349931827204

# Predict
ens0_pred = ens0.predict(df_test) # LB: -.89

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens0_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_adaboostregressor.csv', index=False)

print("Ensemble Model 1: BaggingRegressor")
ens1  = BaggingRegressor(DecisionTreeRegressor(max_depth=4), random_state=1337) #, learning_rate=.05, n_estimators=300,

ens1.fit(df, train.y)

# In Sample R2
ens1_insample_pred = ens1.predict(df)
r2_score(train.y, ens1_insample_pred ) # 0.6998279121628439

# Predict
ens1_pred = ens1.predict(df_test) # LB: -0.77554

submission.y       = ens1_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_baggingreg.csv', index=False)

print("Ensemble Model 2: ExtraTreesRegressor")
ens2  = ExtraTreesRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                            max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, oob_score=False,
                            n_jobs=1, random_state=None, verbose=0, warm_start=False)

ens2.fit(df, train.y)

# In Sample R2
ens2_insample_pred = ens2.predict(df)
r2_score(train.y, ens2_insample_pred ) # 0.97142336381583683

# Predict
ens2_pred = ens2.predict(df_test) # LB: -1.05495

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens2_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_extratrees.csv', index=False)

print("Ensemble Model 3: GradientBoostingRegressor")
ens3  = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                  criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07,
                                  init=None, random_state=None, max_features=None, alpha=0.9, verbose=0,
                                  max_leaf_nodes=None, warm_start=False, presort='auto')

ens3.fit(df, train.y)

# In Sample R2
ens3_insample_pred = ens3.predict(df)
r2_score(train.y, ens3_insample_pred)  # 0.70266651298615024

# Predict
ens3_pred = ens3.predict(df_test)  # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens3_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_gbreg.csv', index=False)

print("Ensemble Model 4: IsolationForest")
ens4  = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, max_features=1.0,
                        bootstrap=False, n_jobs=1, random_state=None, verbose=0)

ens4.fit(df, train.y)

# In Sample R2
ens4_insample_pred = ens4.predict(df)
print(r2_score(train.y, ens4_insample_pred )) #

# Predict
ens4_pred = ens4.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens4_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_isolationforest.csv', index=False)

print("Ensemble Model 5: RandomTreesEmbedding")
""""
ens5  = RandomTreesEmbedding(n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_split=1e-07, 
                             sparse_output=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

ens5.fit(df, train.y)

# In Sample R2
ens5_insample_pred = ens5.predict(df)
print(r2_score(train.y, ens5_insample_pred )) #

# Predict
ens5_pred = ens5.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens5_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_randomtrees.csv', index=False)
"""

print("Ensemble Model 6: RandomForestRegressor")
ens6  = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2,
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                             max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1,
                             random_state=None, verbose=0, warm_start=False)

ens6.fit(df, train.y)

# In Sample R2
ens6_insample_pred = ens6.predict(df)
print(r2_score(train.y, ens6_insample_pred ))  # 0.900319568139

# Predict
ens6_pred = ens6.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens6_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_rf.csv', index=False)

print("Ensemble Model 7: neural_network.MLPRegressor")
ens7  = neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001,
                             batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                             max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                             momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                             beta_1=0.9, beta_2=0.999, epsilon=1e-08)

ens7.fit(df, train.y)

# In Sample R2
ens7_insample_pred = ens7.predict(df)
print(r2_score(train.y, ens7_insample_pred ))  #

# Predict
ens7_pred = ens7.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens7_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_nn_mlp.csv', index=False)

print("Ensemble Model 8: neural_network.BernoulliRBM(")
ens8  = neural_network.BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0,
                                    random_state=None)

ens8.fit(df, train.y)

# In Sample R2
ens8_insample_pred = ens8.predict(df)
print(r2_score(train.y, ens8_insample_pred))  #

# Predict
ens8_pred = ens8.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens8_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_nn_rbm.csv', index=False)

print("Ensemble Model 9: XGB")
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 4,
    'subsample': .921,  # 0.95, # .921
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train.y)
,  # base prediction = mean(target)
    'silent': 1,
    'booster': 'dart',
    'lambda': 1,  # L2 regularization; higher = more conservative
    'alpha': 0  # L1 regularization; higher = more conservative
    # ,'tree_method': 'exact', # Choices: {'auto', 'exact', 'approx', 'hist'}
    # 'grow_policy': 'lossguide' # only works with hist tree_method, Choices: {'depthwise', 'lossguide'}
    # 'normalize_type': 'forest', # DART only; tree or forest, default = "tree"
    # 'rate_drop': 0.1 #[default=0.0] range 0,1
}
n_splits = 5
kf = KFold(n_splits=n_splits)
X = df.as_matrix()
S = df_test.as_matrix()
y = train.y

kf.get_n_splits(X)

X_train, X_valid = X[0:4000, :], X[4001:4208, :]
y_train, y_valid = y[0:4000], y[4001:4208]

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
d_test = xgb.DMatrix(S)
d_train_all = xgb.DMatrix(X, label=y_train)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

model = xgb.train(xgb_params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True,
                  verbose_eval=False)

oos_predictions = model.predict(d_valid, ntree_limit=model.best_ntree_limit)
r2_score(y_valid, oos_predictions ) #

ens9_pred = model.predict(d_test, ntree_limit=model.best_ntree_limit)

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens9_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_xgb.csv', index=False)

print("Ensemble Model 10: LGB")

print("Ensemble Model 11: OLS")
ens11 =linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

ens11.fit(df, train.y)

# In Sample R2
ens11_insample_pred = ens11.predict(df)
r2_score(train.y, ens11_insample_pred ) #

# Predict
ens11_pred = ens11.predict(df_test) # LB:

submission         = pd.read_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/input/sample_submission.csv')
submission.y       = ens11_pred
submission.id      = id
submission.columns = ['ID', 'y']
submission.to_csv('T:/RNA/Baltimore/Jason/ad_hoc/mb/output/layer2_nn_mlp.csv', index=False)

print("Final model stack...")
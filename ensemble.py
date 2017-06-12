# Jason D. Miller
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
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

# Fix duplicate colnames
def maybe_dedup_names(names):
    names = list(names)  # so we can index
    counts = {}

    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)

        if cur_count > 0:
            names[i] = '%s.%d' % (col, cur_count)

        counts[col] = cur_count + 1

    return names

cols=pd.Series(df.columns)
df.columns = maybe_dedup_names(names=cols)

# Create test data
del(files)
os.chdir("T:/RNA/Baltimore/Jason/ad_hoc/mb/layer1_test")
files = glob.glob('*.csv')
dfs_test = {}

# list input files
for f in files:
    dfs_test[os.path.splitext(os.path.basename(f))[0]] = pd.read_csv(f)

# build test data set
for c in dfs.keys():
    pred = dfs_test[c].iloc[:, [1]]
    df_test   = pd.concat([df, pred], axis=1)

df_test = df_test.drop('ID', axis = 1) # this ID doesn't correspond to the training data anyway

# Fix duplicate colnames
def maybe_dedup_names(names):
    names = list(names)  # so we can index
    counts = {}

    for i, col in enumerate(names):
        cur_count = counts.get(col, 0)

        if cur_count > 0:
            names[i] = '%s.%d' % (col, cur_count)

        counts[col] = cur_count + 1

    return names

cols=pd.Series(df_test.columns)
df_test.columns = maybe_dedup_names(names=cols)











# Drop ridiculously bad data
d = df.describe()
for c in d.columns:
    if d[c]['min'] < -100 or d[c]['max'] > 1000:
        df = df.drop(c, axis=1)
df.shape

print("Ensemble Model 0: AdaBoostRegressor")
ens0 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), learning_rate=.05,
                          n_estimators=300, random_state=1337)

ens0.fit(df, train.y)

# Predict
ens0_pred = ens0.predict(test) # LB:

print("Ensemble Model 1: BaggingRegressor")
print("Ensemble Model 2: ExtraTreesRegressor")
print("Ensemble Model 3: GradientBoostingRegressor")
print("Ensemble Model 4: IsolationForest")
print("Ensemble Model 5: RandomTreesEmbedding")
print("Ensemble Model 6: RandomForestRegressor")
print("Ensemble Model 7: neural_network.MLPRegressor")
print("Ensemble Model 8: neural_network.BernoulliRBM(")
print("Ensemble Model 9: XGB")
print("Ensemble Model 10: LGB")
print("Ensemble Model 11: OLS")

print("Final model stack...")
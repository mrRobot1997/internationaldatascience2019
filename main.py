import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape)
print(test.shape)

train.head()
test.head()

all_data = pd.concat((train.loc[:,'Open':'lag_return_96'],
                      test.loc[:,'Open':'lag_return_96']))
all_data.head()

all_data = all_data.drop(['Volume', 'upper_tail','lower_tail'], axis = 1)

cat_feat = ['hour', 'min', 'dayofweek']

cat_data = all_data[cat_feat]

all_data = all_data.drop(cat_feat, axis = 1)
all_data.head()

numeric_feats = all_data.columns
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# print(skewed_feats)
skewed_feats = skewed_feats[skewed_feats > 0.75]
# print(skewed_feats)
# all_data = all_data.
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data.head()

all_data = pd.concat([all_data,cat_data], axis = 1)
all_data.head()

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.up_down

# dayofweek = X_train['dayofweek']
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.2, random_state = 8, shuffle = False)

import lightgbm as lgb

train_data = lgb.Dataset(x_train, y_train, free_raw_data=False, categorical_feature = cat_feat)
valid_data = lgb.Dataset(x_valid, y_valid, free_raw_data=False, categorical_feature = cat_feat)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}

from lightgbm import LGBMClassifier
num_round = 1000
lgbm = LGBMClassifier(num_leaves= 180, max_depth= -1, n_estimators = 2000, n_jobs = 16, random_state = 4, subsample = 0.9, gpu_id = 0, colsample_bytree = 0.85, max_bin = 512, tree_method = 'gpu_hist')
lgbm.fit(X=x_train,y=y_train,eval_set = [(x_train,y_train),(x_valid, y_valid)], eval_metric = ['binary_logloss'], early_stopping_rounds = 70)
# model = lgb.train(parameter, train_data, num_round, valid_sets = [train_data, valid_data], verbose_eval = 100, early_stopping_rounds = 50)

pred_lgb = lgbm.predict(X_test)

idx = []
for i in range (X_test.shape[0]):
    idx.append(i)

mysubmit = pd.DataFrame({'id': idx, 'up_down': pred_lgb})
mysubmit.to_csv('submission.csv', index=True)
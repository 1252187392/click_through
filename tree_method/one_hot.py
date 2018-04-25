#encoding:utf-8
from xgboost import XGBClassifier
import numpy as np
import sys
sys.path.append('.')
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from config import *
from utils import *


mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    assert mode in ['hash','woe']
if mode == 'hash':
    train_features,test_features, train_lables, test_lables = load_hash_data()
else:
    woe_hash_index = np.load(WOE_HASH_INDEX)
    train_features, test_features, train_lables, test_lables = load_woe_data(woe_hash_index)


xgb = joblib.load('./models/{}_xgb.pkl'.format(mode))

values = []
maxx = -1
minn = 100

for feature, label in data_loader(train_features,train_lables,512):
    leves = xgb.apply(feature)
    for i in range(leves.shape[1]):
        col_value = set(leves[:, i])
        if i >= len(values):
            values.append(col_value)
        else:
            values[i] = values[i] | col_value
for i in range(len(values)):
    values[i] = list(values[i])
    maxx = max(maxx, len(values[i]))

del train_features
del train_lables
del test_features
del test_lables

print maxx
print maxx*leves.shape[1]
for i in range(len(values)):
    while len(values[i]) < maxx:
        values[i].append(values[i][-1])
values = np.array(values).T
print values.shape
encoder = OneHotEncoder()
encoder.fit(values)
joblib.dump(encoder,'./models/{}_one_hot_encoder.pkl'.format(mode))

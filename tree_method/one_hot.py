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

def data_loader(features,labels,batch_size):
    for i in range(0, features.shape[0],batch_size):
        left = min(i+batch_size,features.shape[0])
        yield features[i:left,:], labels[i:left]

xgb = joblib.load('./models/woe_xgb.pkl')
train_features, test_features, train_lables, test_labels = load_woe_data([6, 18, 20])

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
del test_labels

print maxx
print maxx*leves.shape[1]
for i in range(len(values)):
    while len(values[i]) < maxx:
        values[i].append(values[i][-1])
values = np.array(values).T
print values.shape
encoder = OneHotEncoder()
encoder.fit(values)
joblib.dump(encoder,'./models/woe_one_hot_encoder.pkl')

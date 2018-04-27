#encoding:utf-8
import sys
sys.path.append('./')
from sklearn.linear_model import LogisticRegression
from utils import *
from sklearn.externals import joblib
import numpy as np
from read_data import *


mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])

assert mode in ['hash','woe']

train_features,test_features, train_lables, test_lables = load_data(mode)

print 'train feature {}'.format(train_features.shape)
print 'test feature {}'.format(test_features.shape)

model = LogisticRegression(penalty='l2', max_iter=200, solver='sag',C= 0.8,random_state=101)
model.fit(train_features, train_lables)
#save model

joblib.dump(model,'models/{}_sk_lr.pkl'.format(mode))

#out train/test auc logloss

y_pred = model.predict_proba(train_features)[:,1]
auc, logloss = get_auc_logloss(train_lables, y_pred)

y_pred = model.predict_proba(test_features)[:,1]
auc, logloss = get_auc_logloss(test_lables, y_pred, 'test')

#make submit csv
del train_features
del train_lables

idx, features, labels = load_pred_data(mode)
y_pred = model.predict_proba(features)[:, 1]
make_submit_csv(idx,y_pred,'submit_csvs/sk_lr_{}.csv'.format(mode))

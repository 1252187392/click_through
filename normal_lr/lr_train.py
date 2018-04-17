#encoding:utf-8
import sys
sys.path.append('./')
from sklearn.linear_model import LogisticRegression
from utils import *
from sklearn.externals import joblib
import numpy as np
from utils import *
from read_data import *


mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    if mode not in ['hash','woe']:
        mode = 'hash'
if mode == 'hash':
    train_features,test_features, train_lables, test_labels = load_hash_data()
else:
    train_features, test_features, train_lables, test_labels = load_woe_data([6, 18, 20])

print 'train feature {}'.format(train_features.shape)
print 'test feature {}'.format(test_features.shape)

model = LogisticRegression(penalty='l2', max_iter=300, solver='sag')
model.fit(train_features, train_lables)
#save model
joblib.dump(model,'models/{}_model.pkl'.format(mode))
#out train/test auc logloss
y_pred = model.predict_proba(train_features)[:,1]
auc, logloss = get_auc_logloss(train_lables, y_pred)

y_pred = model.predict_proba(test_features)[:,1]
auc,logloss = get_auc_logloss(test_labels, y_pred, 'test')
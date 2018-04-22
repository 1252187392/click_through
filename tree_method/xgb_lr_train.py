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
    train_features,test_features, train_lables, test_labels = load_hash_data()
else:
    train_features, test_features, train_lables, test_labels = load_woe_data([6, 18, 20])

print 'feature for train xgb',train_features.shape
#train&save xgb models

xgb = XGBClassifier(max_depth=3,n_estimators=100,gamma=0.1,n_jos = -1,random_state=101,)
xgb.fit(train_features,train_lables,verbose=True)
joblib.dump(xgb,'./models/{}_xgb.pkl'.format(mode))

y_pred = xgb.predict_proba(train_features)[:,1]
auc,loss = get_auc_logloss(train_lables, y_pred)

y_pred = xgb.predict_proba(test_features)[:,1]
auc,loss = get_auc_logloss(test_labels, y_pred,'test')

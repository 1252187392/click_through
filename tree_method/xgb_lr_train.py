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
xgb = XGBClassifier(max_depth=3,n_estimators=19,gamma=0.1,n_jos = -1,random_state=101,)
xgb.fit(train_features,train_lables,verbose=True)
joblib.dump(xgb,'./models/{}_xgb.pkl'.format(mode))

y_pred = xgb.predict_proba(train_features)[:,1]
auc,loss = get_auc_logloss(train_lables, y_pred)

y_pred = xgb.predict_proba(test_features)[:,1]
auc,loss = get_auc_logloss(test_labels, y_pred,'test')

#use one-hot
print 'train one_hot encoder'
leves = xgb.apply(train_features)

encoder = OneHotEncoder()
encoder.fit(leves)
joblib.dump(encoder,'./models/{}_one_hot_encoder.pkl'.format(mode))

onehot = encoder.transform(leves).toarray()

train_features = np.hstack((train_features, onehot))
print 'features for train lr',train_features.shape

#train lr xgb with origin_feature and one-hot feature
lr_model = LogisticRegression(penalty='l1',max_iter=200)
lr_model.fit(train_features, train_lables)

y_pred = lr_model.predict_proba(train_features)[:,1]
acu,loss = get_auc_logloss(train_lables, y_pred)

leves = xgb.apply(test_features)
onehot = encoder.transform(leves).toarray()
test_features = np.hstack((test_features,onehot))
y_pred = lr_model.predict_proba(test_features)[:,1]
auc,loss =get_auc_logloss(test_labels, y_pred,'test')

joblib.dump(lr_model,'./models/{}_xgb_lr.pkl'.format(mode))

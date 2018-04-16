from xgboost import XGBClassifier
import numpy as np
import sys
sys.path.append('.')
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from config import *
from utils import load_woe_data


train_feature,test_feature,train_labels,test_labels = load_woe_data([6, 18, 20])
model = XGBClassifier(max_depth=3,n_estimators=100,gamma=0.1,n_jos = -1,random_state=101,)

model.fit(train_feature,train_labels,verbose=True)

y_pred = model.predict_proba(train_feature)[:,1]
auc = roc_auc_score(train_labels, y_pred)
logloss = log_loss(train_labels, y_pred)
print 'train auc:{},log loss:{}'.format(auc, logloss)

y_pred = model.predict_proba(test_feature)[:,1]
auc = roc_auc_score(test_labels, y_pred)
logloss = log_loss(test_labels, y_pred)
print 'train auc:{},log loss:{}'.format(auc, logloss)
leves = model.apply(train_feature)
#print leves.shape
#print leves[0]

encoder = OneHotEncoder()
encoder.fit(leves)

#print x
#xx = np.array([[1,2,3]])
#x = np.hstack((xx,x))
#print x

onehot = encoder.transform(leves).toarray()
print onehot.shape

train_feature = np.hstack((train_feature, onehot))
print train_feature.shape
lr_model = LogisticRegression(penalty='l1',max_iter=200)
lr_model.fit(train_feature, train_labels)


y_pred = lr_model.predict_proba(train_feature)[:,1]
auc = roc_auc_score(train_labels, y_pred)
logloss = log_loss(train_labels, y_pred)
print 'train auc:{},log loss:{}'.format(auc, logloss)

leves = model.apply(test_feature)
onehot = encoder.transform(leves).toarray()
test_feature = np.hstack((test_feature,onehot))
y_pred = lr_model.predict_proba(test_feature)[:,1]
auc = roc_auc_score(test_labels, y_pred)
logloss = log_loss(test_labels, y_pred)
print 'train auc:{},log loss:{}'.format(auc, logloss)

joblib.dump(lr_model,'./models/xgb.pkl')
joblib.dump(model,'./models/xgb_lr.pkl')
joblib.dump(encoder,'./models/one_hot_encoder.pkl')
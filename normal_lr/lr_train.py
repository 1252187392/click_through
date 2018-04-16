#encoding:utf-8
import sys
sys.path.append('./')
from sklearn.linear_model import LogisticRegression, SGDClassifier
from utils import *
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from csv import DictReader
import numpy as np



def load_woe_data():
    train_features = np.load('cache_data/woe_train_features.npy')
    train_labels = np.load('cache_data/woe_train_labels.npy')

    test_features = np.load('cache_data/woe_test_features.npy')
    test_labels = np.load('cache_data/woe_test_labels.npy')

    train_features = np.delete(train_features, [6, 18, 20], axis=1)
    test_features = np.delete(test_features, [6, 18, 20], axis=1)
    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)

model = LogisticRegression(penalty='l2', max_iter=300, solver='sag')
mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    if mode not in ['hash','woe']:
        mode = 'hash'
if mode == 'hash':
    features,test_features, labels, test_labels = load_hash_data()
else:
    features, test_features, labels, test_labels = load_woe_data()
print test_features.shape
model.fit(features, labels)
y_pred = model.predict_proba(features)[:,1]
auc = roc_auc_score(labels, y_pred)
logloss = log_loss(labels, y_pred)
print 'train auc:{},log loss:{}'.format(auc, logloss)
joblib.dump(model,'models/{}_model.pkl'.format(mode))

y_pred = model.predict_proba(test_features)[:,1]
auc = roc_auc_score(test_labels, y_pred)
logloss = log_loss(test_labels, y_pred)
print 'test auc:{},log loss:{}'.format(auc, logloss)
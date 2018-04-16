#encoding:utf-8
from sklearn.linear_model import LogisticRegression, SGDClassifier
from utils import *
from sklearn.metrics import log_loss,roc_auc_score
from csv import DictReader
import numpy as np

def get_batch_data(filename,batch_size):
    features = []
    labels = []
    cnt = 0
    D = 2**20
    fin = open(filename)
    for row in DictReader(fin):
        ID,feature,label = hash_data(cnt,row,D)
        features.append(feature)
        labels.append(label)
        cnt += 1
        if cnt % batch_size == 0:
            yield features,labels
            features = []
            labels = []

filename = 'origin_datas/part_train.csv'
model = LogisticRegression(penalty='l2',max_iter=1)
model = SGDClassifier(penalty='l2',loss='log')
epochs = 10
batch_size = 512

for e in range(epochs):
    batch_loader = get_batch_data(filename, batch_size)
    cnt = 0
    for features,labels in batch_loader:
        cnt += 1
        #if cnt % 8 == 0:
        #    print cnt, len(labels),features[0]
        model = model.partial_fit(features,labels,np.unique(labels))
    print 'epochs', e
    y_pred = []
    labels = []
    batch_loader = get_batch_data(filename, batch_size)
    for features, _labels in batch_loader:
        pred = model.predict_proba(features)[:,1].tolist()
        y_pred.extend(pred)
        labels.extend(_labels)
    #print labels[:10]
    #print y_pred[:10]
    auc = roc_auc_score(labels, y_pred)
    logloss = log_loss(labels, y_pred)
    print 'auc:{},log loss:{}'.format(auc, logloss)
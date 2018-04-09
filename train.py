#encoding:utf-8
from sklearn.linear_model import LogisticRegression
from utils import *


def get_batch_data(loader,batch_size):
    features = []
    labels = []
    cnt = 0
    for ID, feautre, label in loader:
        features.append(feautre)
        labels.append(label)
        cnt += 1
        if cnt % batch_size == 0:
            yield features,labels
            features = []
            labels = []

filename = 'datas/train_part.csv'
model = LogisticRegression(penalty='l2',max_iter=1)
epochs = 10
batch_size = 512

for e in range(epochs):
    loader = hash_data_loader(filename, 2**20)
    batch_loader = get_batch_data(loader,batch_size)
    cnt = 0
    for features,labels in batch_loader:
        cnt += 1
        if cnt % 200 == 0:
            print cnt, len(labels)
        model.fit(features,labels)
    print 'epochs', e
    y_pred = []
    labels = []
    loader = hash_data_loader(filename, 2 ** 20)
    batch_loader = get_batch_data(loader, batch_size)
    for features, _labels in batch_loader:
        pred = model.predict_proba(features)[:,1].tolist()
        y_pred.extend(pred)
        labels.extend(_labels)

    auc = roc_auc_score(labels, y_pred)
    logloss = log_loss(labels, y_pred)
    print 'auc:{},log loss:{}'.format(auc, logloss)
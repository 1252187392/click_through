from sklearn.externals import joblib
import sys
from read_data import clean_data_by_woe
from sklearn.metrics import log_loss,roc_auc_score
import numpy as np
from csv import DictWriter
mode = 'hash'
if len(sys.argv) > 1:
    mode = sys.argv[1]
model = joblib.load('models/{}_model.pkl'.format(mode))

test_filename = 'origin_datas/part_test.csv'
#test_filename = 'origin_datas/part_train.csv'
idx, features, labels = clean_data_by_woe(test_filename)
features = np.delete(features,[6,18,20],axis=1)
print features[0]
y_pred = model.predict_proba(features)[:,1]
auc = roc_auc_score(labels, y_pred)
logloss = log_loss(labels, y_pred)
print 'test auc:{},log loss:{}'.format(auc, logloss)
fout = open('origin_datas/submit.csv','w')
writer = DictWriter(fout,fieldnames=['id','click'])
writer.writeheader()
for ids, click in zip(idx, y_pred):
    row = {'id':ids,'click':click}
    writer.writerow(row)
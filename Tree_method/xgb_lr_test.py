import sys
sys.path.append('.')
from read_data import clean_data_by_woe,clean_data_by_hash
from sklearn.externals import joblib
import numpy as np
from config import *

mode = 'hash'
if len(sys.argv) > 1:
    mode = sys.argv[1]
    assert mode in ['hash','woe']

xgb = joblib.load('./models/{}_xgb.pkl'.format(mode))
lr = joblib.load('./models/{}_xgb_lr.pkl'.format(mode))
encoder = joblib.load('./models/{}_one_hot_encoder.pkl'.format(mode))
if mode == 'woe':
    idx, features, labels = clean_data_by_woe(ORIGIN_TEST_FILE)
    features = np.delete(features,[6,18,20],axis=1)
else:
    idx, features, labels = clean_data_by_hash(ORIGIN_TEST_FILE)
leves = xgb.apply(features)
onehot = encoder.transform(leves).toarray()
features = np.hstack((features,onehot))
y_pred = lr.predict_proba(features)[:,1]

make_submit_csv(idx, y_pred, 'submit_csvs/{}_xgb_lr.csv'.format(mode))
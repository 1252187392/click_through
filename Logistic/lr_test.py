import sys
sys.path.append('.')
from sklearn.externals import joblib
from read_data import clean_data_by_woe
import numpy as np
from utils import make_submit_csv
from config import *

mode = 'hash'
if len(sys.argv) > 1:
    mode = sys.argv[1]
    assert mode in ['hash','woe']
model = joblib.load('models/{}_model.pkl'.format(mode))

idx, features, labels = clean_data_by_woe(ORIGIN_TEST_FILE)
features = np.delete(features,[6,18,20],axis=1)
y_pred = model.predict_proba(features)[:,1]
make_submit_csv(idx, y_pred, 'submit_csvs/{}_lr.csv'.format(mode))

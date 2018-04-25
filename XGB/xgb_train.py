from xgboost import XGBClassifier
import numpy as np
import sys
sys.path.append('.')
from sklearn.externals import joblib
from config import *
from utils import *
from read_data import *

mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
assert mode in ['hash','woe']

train_features,test_features, train_lables, test_labels = load_data(mode)

print 'feature for train xgb',train_features.shape

#train&save xgb models

print test_lables
xgb = XGBClassifier(max_depth=4,n_estimators=120,gamma=0.1,
                    n_jos = -1,random_state=101,learning_rate=0.1)
xgb.fit(train_features,train_lables,verbose=True,
        eval_set=[(test_features,test_lables)],eval_metric='logloss')
joblib.dump(xgb,'./models/{}_xgb.pkl'.format(mode))

y_pred = xgb.predict_proba(train_features)[:,1]
auc,loss = get_auc_logloss(train_lables, y_pred)

y_pred = xgb.predict_proba(test_features)[:,1]
auc,loss = get_auc_logloss(test_lables, y_pred,'test')
print test_features[:2]

del train_features
del train_lables
if mode == 'woe':
    woe_hash_index = np.load(WOE_HASH_INDEX)
    idx, features, labels = clean_data_by_woe(ORIGIN_TEST_FILE)
    features = np.delete(features, woe_hash_index, axis=1)
else:
    idx, features, labels = clean_data_by_hash(ORIGIN_TEST_FILE)

y_pred = xgb.predict_proba(features)[:,1]
make_submit_csv(idx, y_pred, 'submit_csvs/{}_xgb.csv'.format(mode))
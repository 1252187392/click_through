from xgboost import XGBClassifier
import sys
sys.path.append('.')
from read_data import *

mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
assert mode in ['hash','woe']

train_features,test_features, train_lables, test_lables = load_data(mode)

print 'feature for train xgb',train_features.shape

#train&save xgb models

xgb = XGBClassifier(max_depth=4,n_estimators=120,gamma=0.1,
                    n_jos = -1,random_state=101,learning_rate=0.01,
                    colsample_bytree=0.8)
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

idx, features, labels = load_pred_data(mode)
y_pred = xgb.predict_proba(features)[:,1]
make_submit_csv(idx, y_pred, 'submit_csvs/{}_xgb.csv'.format(mode))

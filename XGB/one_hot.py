#encoding:utf-8
import sys
sys.path.append('.')
from sklearn.preprocessing import OneHotEncoder
from read_data import *

mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    assert mode in ['hash','woe']

train_features,test_features, train_lables, test_lables = load_data(mode)

xgb = joblib.load('./models/{}_xgb.pkl'.format(mode))

values = []
maxx = -1
minn = 100

for feature, label in data_loader(train_features,train_lables,512):
    leves = xgb.apply(feature)
    for i in range(leves.shape[1]):
        col_value = set(leves[:, i])
        if i >= len(values):
            values.append(col_value)
        else:
            values[i] = values[i] | col_value
for i in range(len(values)):
    values[i] = list(values[i])
    maxx = max(maxx, len(values[i]))

del train_features
del train_lables
del test_features
del test_lables

print maxx
print maxx*leves.shape[1]
for i in range(len(values)):
    while len(values[i]) < maxx:
        values[i].append(values[i][-1])
values = np.array(values).T
print values.shape
encoder = OneHotEncoder()
encoder.fit(values)
joblib.dump(encoder,'./models/{}_one_hot_encoder.pkl'.format(mode))

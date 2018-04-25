#encoding:utf-8

from csv import DictReader, DictWriter
from math import log
import numpy as np
from sklearn.metrics import log_loss,roc_auc_score,precision_score
from sklearn.model_selection import train_test_split
from config import *
from datetime import datetime

def load_data(mode):
    if mode == 'hash':
        train_features, test_features, train_lables, test_lables = load_hash_data()
    else:
        woe_hash_index = np.load(WOE_HASH_INDEX)
        train_features, test_features, train_lables, test_lables = load_woe_data(woe_hash_index)
    return train_features, test_features, train_lables, test_lables

def data_loader(features,labels,batch_size):
    for i in range(0, features.shape[0],batch_size):
        left = min(i+batch_size,features.shape[0])
        yield features[i:left,:], labels[i:left]

def hash_data(row, D):
    '''
    对一行data进行hash处理
    :param row: 原始数据
    :param D: mod
    :return:
    '''
    ID = row['id']
    del row['id']
    y = 0
    if 'click' in row:
        y = int(row['click'])
        del row['click']
    date = int(row['hour'][4:6])
    # turn hour really into hour, it was originally YYMMDDHH
    row['hour'] = row['hour'][6:]
    x = []
    for key in row:
        value = row[key]
        index = abs(hash(key + '_' + value)) % D
        x.append(index)
    return ID, x, y

def cal_woe_dict(values,labels):
    '''
    计算每个value值对应的woe
    woe = log(p(pos)/p(neg))
    :param values:
    :param labels:
    :return:{value:woe}
    '''
    values_info = {}
    pos_total = sum(labels) + 1
    neg_total = len(labels) - pos_total + 2
    for value, label in zip(values, labels):
        if value not in values_info:
            values_info[value] = [0,0]
        values_info[value][int(label)] += 1
    woe_dict = {}
    sum_woe = 0
    for value, cnt_pair in values_info.iteritems():
        cnt_pair[0] += 1
        cnt_pair[1] += 1
        proba_pair = [cnt_pair[0]*1.0/neg_total,cnt_pair[1]*1.0/pos_total]
        #print proba_pair
        woe = log(proba_pair[1]/proba_pair[0])
        woe_dict[value] = woe
        sum_woe += woe
    woe_dict['default'] = sum_woe / len(woe_dict)
    return woe_dict
#data_loader('datas/train_part.csv')

def load_hash_data():
    train_features = np.load(HASH_TRAIN_FEATURE)
    train_labels = np.load(HASH_TRAIN_LABLE)

    test_features = np.load(HASH_TEST_FEATURE)
    test_labels = np.load(HASH_TEST_LABLE)
    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)


def load_woe_data(delete_list = []):
    #[6, 18, 20]
    train_features = np.load(WOE_TRAIN_FEATURE)
    train_labels = np.load(WOE_TRAIN_LABLE)

    test_features = np.load(WOE_TEST_FEATURE)
    test_labels = np.load(WOE_TEST_LABLE)

    train_features = np.delete(train_features, delete_list, axis=1)
    test_features = np.delete(test_features, delete_list, axis=1)
    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)

def get_auc_logloss(y_true, y_pred, info='train'):
    '''
    计算auc logloss，并打印
    :param y_true:
    :param y_pred:
    :param info: train/test
    :return: auc,logloss
    '''
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    click = [1 if _ >= 0.5 else 0 for _ in y_pred]
    #p = precision_score(y_true, [1 if _ >= 0.5 else 0 for _ in y_pred])
    x = [1 if click[i] == y_true[i] else 0 for i in range(len(click))]
    p = sum(x) * 1.0 / len(y_true)
    print '{} auc:{},logloss:{},precision:{}'.format(info, auc, logloss, p)
    return auc, logloss

def count_time(func):
    '''
    计时器
    :param func:
    :return:
    '''
    def wrapper(*args,**kwargs):
        print '{},begin {}'.format(datetime.now(),func.__name__)
        x = func(*args,**kwargs)
        print '{},finish {}'.format(datetime.now(),func.__name__)
        return x
    return wrapper

def make_submit_csv(idx, scores, filename):
    '''
    生成用于提交的csv文件，数据格式id,click_proba
    :param idx:
    :param scores:
    :param filename:
    :return:None
    '''
    now = datetime.now()
    nowstr = str(now.day)+str(now.hour)+str(now.minute)+'.csv'
    fout = open(filename.replace('.csv',nowstr), 'w')
    writer = DictWriter(fout, fieldnames=['id', 'click'])
    writer.writeheader()
    for Id, score in zip(idx,scores):
        row = {'id':Id, 'click':score}
        writer.writerow(row)
    fout.close()
    print 'score files ' + filename.replace('.csv',nowstr)

if __name__ == '__main__':
    test_woe = [[1,1,2,2,3,3],[1,0,1,0,1,0]]
    print cal_woe_dict(test_woe[0],test_woe[1])

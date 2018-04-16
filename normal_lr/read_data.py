#encoding:utf-8
import sys
import os
sys.path.append('.')
from utils import *
from csv import DictReader, DictWriter
import numpy as np
import random
from sklearn.externals import joblib

TRAIN_IDX_PATH = 'cache_data/woe_dict_files/train_id.npy'
TEST_IDX_PATH = 'cache_data/woe_dict_files/test_idx.npy'

def clean_data_by_hash(filename):
    features = []
    labels = []
    idx = []
    D = 2**20
    fin = open(filename)
    cnt = 0
    for row in DictReader(fin):
        #return row.keys()
        ID,feature,label = hash_data(cnt,row,D)
        idx.append(ID)
        features.append(feature)
        labels.append(label)
        cnt += 1
    fin.close()
    return np.array(idx), np.array(features), np.array(labels)

def processing_woe_info(filename):
    COL_NAME = ['C21', 'device_ip', 'site_id', 'app_id', 'C19', 'C18',\
                'device_type', 'C17', 'C15', 'C14', 'C16',\
                'device_conn_type', 'C1', 'app_category', 'site_category',\
                'app_domain', 'site_domain', 'banner_pos', 'device_id', 'C20',\
                'hour', 'device_model']
    idx = []
    with open(filename) as fin:
        for row in DictReader(fin):
            ID = row['id']
            idx.append(ID)
    random.shuffle(idx)
    #print len(idx)
    cut = int(len(idx) * 0.95)
    train_idx, test_idx = idx[:cut], idx[cut:]
    train_file = 'cache_data/woe_dict_files/woe_train.csv'
    test_file = 'cache_data/woe_dict_files/woe_test.csv'
    fin = open(filename)
    for row in DictReader(fin):
        keys = row.keys()
        break
    train_writer = DictWriter(open(train_file, 'w'),fieldnames=keys)
    test_writer = DictWriter(open(test_file, 'w'),fieldnames=keys)
    train_writer.writeheader()
    test_writer.writeheader()
    np.save(TRAIN_IDX_PATH, train_idx)
    np.save(TEST_IDX_PATH, test_idx)
    train_set = set(train_idx)
    del idx
    write_flag = True
    for col in COL_NAME:
        if col == 'device_ip' or col == 'device_id' or col == 'device_model':
            continue
        fin = open(filename)
        values, labels = [], []
        for row in DictReader(fin):
            ID, value, label = row['id'], row[col], int(row['click'])
            if col == 'hour':
                value = value[6:]
            if ID not in train_set:
                if write_flag:
                    test_writer.writerow(row)
                continue
            if write_flag:
                train_writer.writerow(row)
            labels.append(label)
            values.append(value)
        fin.close()
        write_flag = False
        print col,len(set(values))
        one_woe_dict = cal_woe_dict(values,labels)
        joblib.dump(one_woe_dict,'cache_data/woe_dict_files/{}_woe.dict'.format(col))

def clean_data_by_woe(filename):
    woe_dict = {}
    woe_path = 'cache_data/woe_dict_files/'
    file_list = os.listdir(woe_path)
    for woe_file in file_list:
        if 'dict' not in woe_file:
            continue
        col_name = woe_file.replace('_woe.dict','')
        one_woe_dict = joblib.load(woe_path+woe_file)
        woe_dict[col_name] = one_woe_dict
    idx, features, labels = [], [], []
    feature_names = ['site_id', 'C20', 'C19', 'site_domain', 'device_type', 'C17', \
                     'device_ip', 'C14', 'C16', 'C15', 'device_conn_type', 'C1', \
                     'app_category', 'site_category', 'app_domain', 'C21', 'banner_pos',\
                     'app_id', 'device_id', 'hour', 'device_model', 'C18']
    first = True
    fin = open(filename)
    for row in DictReader(fin):
        idx.append(row['id'])
        del row['id']
        if 'click' in row:
            labels.append(int(row['click']))
            del row['click']
        else:
            labels.append(random.randint(0,1))
        feature = []
        for key in feature_names:
            #print key
            value = row[key]
            if key == 'hour':
                value = value[6:]
            #print key,value
            if key in woe_dict:
                if value in woe_dict[key]:
                    woe = woe_dict[key][value]
                else:
                    woe = woe_dict[key]['default']
            else:
                woe = hash(value)
            #if first:
            #    feature_names.append(key)
            feature.append(woe)
        first = False
        features.append(feature)
    print feature_names
    return np.array(idx), np.array(features), np.array(labels)

if __name__ == '__main__':
    filename = 'origin_datas/part_train.csv'
    idx, features, labels = clean_data_by_hash(filename)
    np.save('cache_data/hash_idx.npy', features)
    np.save('cache_data/hash_features.npy', features)
    np.save('cache_data/hash_labels.npy', labels)
    del features
    del labels
    processing_woe_info(filename)
    train_file = 'cache_data/woe_dict_files/woe_train.csv'
    test_file = 'cache_data/woe_dict_files/woe_test.csv'
    woe_idx, woe_feature, woe_label = clean_data_by_woe(train_file)
    np.save('cache_data/woe_train_idx.npy', woe_idx)
    np.save('cache_data/woe_train_features.npy', woe_feature)
    np.save('cache_data/woe_train_labels.npy', woe_label)

    woe_idx, woe_feature, woe_label = clean_data_by_woe(test_file)
    np.save('cache_data/woe_test_idx.npy', woe_idx)
    np.save('cache_data/woe_test_features.npy', woe_feature)
    np.save('cache_data/woe_test_labels.npy', woe_label)

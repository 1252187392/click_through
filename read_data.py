#encoding:utf-8
import sys
import os
sys.path.append('.')
from utils import *
from csv import DictReader, DictWriter
import numpy as np
import random
from sklearn.externals import joblib
from config import *
from multiprocessing import Pool

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

def make_one_woe(values,labels,col):
    one_woe_dict = cal_woe_dict(values, labels)
    joblib.dump(one_woe_dict, 'cache_data/woe_dict_files/{}_woe.dict'.format(col))

def processing_woe_info(filename):
    '''
    计算特征对应的woe值


    :param filename:
    :return:
    '''
    idx = []
    with open(filename) as fin:
        for row in DictReader(fin):
            ID = row['id']
            idx.append(ID)
    random.shuffle(idx)
    cut = int(len(idx) * 0.95)
    train_idx, test_idx = idx[:cut], idx[cut:]

    fin = open(filename)
    for row in DictReader(fin):
        keys = row.keys()
        break
    train_writer = DictWriter(open(WOE_TRAIN_FILE, 'w'),fieldnames=keys)
    test_writer = DictWriter(open(WOE_TEST_FILE, 'w'),fieldnames=keys)
    train_writer.writeheader()
    test_writer.writeheader()
    #np.save(TRAIN_IDX_PATH, train_idx)
    #np.save(TEST_IDX_PATH, test_idx)
    train_set = set(train_idx)
    del idx
    pool = Pool(processes=4)
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
        pool.apply_async(make_one_woe,args=(values,labels,col))
    pool.close()
    pool.join()


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
        for key in FEATURE_NAME:
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
            feature.append(woe)
        features.append(feature)
    return np.array(idx), np.array(features), np.array(labels)

def full_mode():
    clean_data_task(ORIGIN_TRAIN_FILE, HASH_SAVE)
    processing_woe_info(ORIGIN_TRAIN_FILE)
    clean_data_task(WOE_TRAIN_FILE, WOE_TRAIN_SAVE, 'woe')
    clean_data_task(WOE_TEST_FILE, WOE_TEST_SAVE, 'woe')

def clean_data_task(filename,save_path,mode='hash'):
    if mode == 'hash':
        idx, features, lables = clean_data_by_hash(filename)
    else:
        idx, features, lables = clean_data_by_woe(filename)
    np.save(save_path+'_features.npy', features)
    np.save(save_path+'_lables.npy', lables)

def split_mode():
    if not os.path.exists(ORIGIN_TEST_FILE.replace('.csv','')):
        os.system('python split_csv.py {} 5000000'.format(ORIGIN_TRAIN_FILE))
    dirname = ORIGIN_TRAIN_FILE.replace('.csv','')
    part_train_files = os.listdir(dirname)
    save_dirname = HASH_FEATURE.replace('.npy','/')
    os.system('mkdir ' + save_dirname)
    pool = Pool(processes=2)
    for part_file in part_train_files:
        if '.csv' not in part_file:
            continue
        save_path = save_dirname+part_file.replace('.csv','')
        pool.apply_async(clean_data_task,(dirname + '/' + part_file, save_path))
    pool.close()
    pool.join()
    processing_woe_info(ORIGIN_TRAIN_FILE)
    if not os.path.exists(WOE_TRAIN_FILE.replace('.csv','')):
        os.system('python split_csv.py {} 5000000'.format(WOE_TRAIN_FILE))
    dirname = WOE_TRAIN_FILE.replace('.csv', '')
    part_train_files = os.listdir(dirname)
    save_dirname = WOE_TRAIN_FEATURE.replace('.npy','/')
    os.system('mkdir ' + save_dirname)
    pool = Pool(processes=2)
    for part_file in part_train_files:
        if '.csv' not in part_file:
            continue
        save_path = save_dirname + part_file.replace('.csv','')
        pool.apply_async(clean_data_task,(dirname + '/' + part_file, save_path,'woe'))
    pool.close()
    pool.join()
    clean_data_task(WOE_TEST_FILE, WOE_TEST_SAVE, 'woe')
if __name__ == '__main__':
    assert len(sys.argv) > 1
    mode = sys.argv[1]
    assert mode in ['full','split']
    if mode == 'full':
        full_mode()
    else:
        split_mode()

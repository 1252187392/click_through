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
from datetime import datetime

def split_train_test_file(filename,rate):
    '''
    spilt origin file to trian/test file
    :param filename:
    :return: None
    '''
    print '{} begin split {},rate {}'.format(datetime.now(),filename,rate)
    idx = []
    keys = []
    with open(filename) as fin:
        for row in DictReader(fin):
            idx.append(row['id'])
        keys = row.keys()
    random.shuffle(idx)
    cut = int(len(idx) * rate)
    test_set = set(idx[cut:])
    train_writer = DictWriter(open(TRAIN_FILE, 'w'), fieldnames=keys)
    test_writer = DictWriter(open(TEST_FILE, 'w'), fieldnames=keys)
    train_writer.writeheader()
    test_writer.writeheader()
    with open(filename) as fin:
        for row in DictReader(fin):
            if row['id'] not in test_set:
                train_writer.writerow(row)
            else:
                test_writer.writerow(row)
    print '{} end split'.format(datetime.now())

def make_one_woe(filename, colname):
    '''
    处理单个特征对应的woe
    :param filename:
    :param colname:
    :return:
    '''
    print '{} begin calculate woe:{}'.format(datetime.now(), colname)
    fin = open(filename)
    values, labels = [], []
    for row in DictReader(fin):
        ID, value, label = row['id'], row[colname], int(row['click'])
        if colname == 'hour':
            value = value[6:]
        labels.append(label)
        values.append(value)
    fin.close()
    value_nums = len(set(values))
    if value_nums * 1.0 / len(values) < 0.5:
        one_woe_dict = cal_woe_dict(values, labels)
        joblib.dump(one_woe_dict, 'cache_data/woe_dict_files/{}_woe.dict'.format(colname))
        print '{} finish calculate {} woe'.format(datetime.now(), colname)
    else:
        print '{},{} values are too many,no need woe'.format(datetime.now(), colname)

@count_time
def processing_woe_info(filename):
    '''
    计算特征对应的woe值
    :param filename:
    :return:
    '''
    pool = Pool(processes=PROCESSES_FOR_WOE)
    for col in COL_NAME:
        if col == 'click' or col == 'id':
            continue
        #if col == 'device_ip' or col == 'device_id' or col == 'device_model':
        #    continue
        pool.apply_async(make_one_woe, args=(filename, col))
    pool.close()
    pool.join()

@count_time
def clean_data_by_woe(filename):
    '''
    使用woe方法处理特征
    :param filename:
    :return:
    '''
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
            if key in woe_dict:
                if value in woe_dict[key]:
                    woe = woe_dict[key][value]
                else:
                    woe = woe_dict[key]['default']
            else:
                woe = hash(value)
            feature.append(woe)
        features.append(feature)
    hash_index = []
    for i,key in enumerate(FEATURE_NAME):
        if key not in woe_dict:
            hash_index.append(i)
    np.save(WOE_HASH_INDEX,hash_index)
    return np.array(idx), np.array(features), np.array(labels)

@count_time
def clean_data_by_hash(filename):
    '''
    使用hash方法获取特征值
    :param filename:
    :return: idx,feature,lable
    '''
    features = []
    labels = []
    idx = []
    D = 2**20
    fin = open(filename)
    cnt = 0
    for row in DictReader(fin):
        ID,feature,label = hash_data(row,D)
        idx.append(ID)
        features.append(feature)
        labels.append(label)
        cnt += 1
    fin.close()
    return np.array(idx), np.array(features), np.array(labels)

def clean_data_task(filename,save_path,mode='hash'):
    '''
    读取文件,并保存对应的feature,lable
    :param filename:
    :param save_path:
    :param mode:
    :return:
    '''
    if mode == 'hash':
        idx, features, lables = clean_data_by_hash(filename)
    else:
        idx, features, lables = clean_data_by_woe(filename)
    np.save(save_path+'_features.npy', features)
    np.save(save_path+'_lables.npy', lables)

def full_mode():
    '''
    对文件直接进行读取
    :return:
    '''
    clean_data_task(ORIGIN_TRAIN_FILE, HASH_SAVE)
    processing_woe_info(ORIGIN_TRAIN_FILE)
    clean_data_task(WOE_TRAIN_FILE, WOE_TRAIN_SAVE, 'woe')
    clean_data_task(WOE_TEST_FILE, WOE_TEST_SAVE, 'woe')

@count_time
def split_mode():
    '''
    对文件先切分,小批量读取
    :return:
    '''
    if not os.path.exists(TRAIN_FILE.replace('.csv','')):
        os.system('python split_csv.py {} 5000000'.format(TRAIN_FILE))
    dirname = TRAIN_FILE.replace('.csv','')
    part_train_files = os.listdir(dirname)
    save_dirname = HASH_TRAIN_FEATURE.replace('.npy','/')
    os.system('mkdir ' + save_dirname)
    pool = Pool(processes=PROCESSES_FOR_CLEAN)
    for part_file in part_train_files:
        if '.csv' not in part_file:
            continue
        save_path = save_dirname+part_file.replace('.csv','')
        pool.apply_async(clean_data_task,(dirname + '/' + part_file, save_path))
    pool.close()
    pool.join()
    clean_data_task(TEST_FILE, HASH_TEST_SAVE)

    processing_woe_info(TRAIN_FILE)
    dirname = TRAIN_FILE.replace('.csv', '')
    part_train_files = os.listdir(dirname)
    save_dirname = WOE_TRAIN_FEATURE.replace('.npy','/')
    os.system('mkdir ' + save_dirname)
    pool = Pool(processes=PROCESSES_FOR_CLEAN)
    for part_file in part_train_files:
        if '.csv' not in part_file:
            continue
        save_path = save_dirname + part_file.replace('.csv','')
        pool.apply_async(clean_data_task,(dirname + '/' + part_file, save_path,'woe'))
    pool.close()
    pool.join()
    clean_data_task(TEST_FILE, WOE_TEST_SAVE, 'woe')
    np.save('cache_data/feature_name.npy',FEATURE_NAME)

if __name__ == '__main__':
    assert len(sys.argv) > 1
    mode = sys.argv[1]
    assert mode in ['full','split']
    split_train_test_file(ORIGIN_TRAIN_FILE, TRAIN_SIZE)
    mode = 'split'
    if mode == 'full':
        full_mode()
    else:
        split_mode()


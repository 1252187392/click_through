#encoding:utf-8

from csv import DictReader
from math import log
COL_NAME = ['C21', 'device_ip', 'site_id', 'app_id', 'C19', 'C18', \
            'device_type', 'id', 'C17', 'click', 'C15', 'C14', 'C16', \
            'device_conn_type', 'C1', 'app_category', 'site_category', \
            'app_domain', 'site_domain', 'banner_pos', 'device_id', 'C20',\
            'hour', 'device_model']

def data_loader(filename):
    with open(filename) as fin:
        reader = DictReader(fin)
        for row in reader:
            print row
            col_name = row.keys()
            print col_name
            break

def hash_data(t,row, D):
    ID = row['id']
    del row['id']
    # process clicks
    y = 0
    if 'click' in row:
        y = int(row['click'])
        del row['click']
    # extract date
    date = int(row['hour'][4:6])

    # turn hour really into hour, it was originally YYMMDDHH
    row['hour'] = row['hour'][6:]

    # build x
    x = []
    for key in row:
        value = row[key]
        # one-hot encode everything with hash trick
        index = abs(hash(key + '_' + value)) % D
        x.append(index)
    #yield t, date, ID, x, y
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

if __name__ == '__main__':
    test_woe = [[1,1,2,2,3,3],[1,0,1,0,1,0]]
    print cal_woe_dict(test_woe[0],test_woe[1])
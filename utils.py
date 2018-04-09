#encoding:utf-8

from csv import DictReader

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

def hash_data_loader(filename, D):
    for t, row in enumerate(DictReader(open(filename))):
        ID = row['id']
        del row['id']
        # process clicks
        y = int(row['click'])
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
        yield ID, x, y


#data_loader('datas/train_part.csv')
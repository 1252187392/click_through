#encoding:utf-8
FEATURE_NAME = ['site_id', 'C20', 'C19', 'site_domain', 'device_type', 'C17', \
                'device_ip', 'C14', 'C16', 'C15', 'device_conn_type', 'C1', \
                'app_category', 'site_category', 'app_domain', 'C21', 'banner_pos',\
                'app_id', 'device_id', 'hour', 'device_model', 'C18']

COL_NAME = ['C21', 'device_ip', 'site_id', 'app_id', 'C19', 'C18', \
            'device_type', 'id', 'C17', 'click', 'C15', 'C14', 'C16', \
            'device_conn_type', 'C1', 'app_category', 'site_category', \
            'app_domain', 'site_domain', 'banner_pos', 'device_id', 'C20',\
            'hour', 'device_model']

#训练集 测试机切分比例
TRAIN_SIZE = 0.95

#读取文件时的进程数
PROCESSES_FOR_CLEAN = 3

#计算woe映射时的进程数
PROCESSES_FOR_WOE = 2

#切分文件时,每PART_SIZE行切为一个文件
PART_SIZE = 1000

ORIGIN_TRAIN_FILE = 'origin_datas/train.csv'
ORIGIN_TEST_FILE = 'origin_datas/test.csv'

TRAIN_FILE = 'cache_data/use_train.csv'
TEST_FILE = 'cache_data/use_test.csv'

WOE_TRAIN_SAVE = 'cache_data/woe_train'
WOE_TRAIN_FEATURE = 'cache_data/woe_train_features.npy'
WOE_TRAIN_LABLE = 'cache_data/woe_train_lables.npy'

WOE_TEST_SAVE = 'cache_data/woe_test'
WOE_TEST_FEATURE = 'cache_data/woe_test_features.npy'
WOE_TEST_LABLE = 'cache_data/woe_test_lables.npy'
WOE_HASH_INDEX = 'cache_data/woe_hash_index.npy'

HASH_TRAIN_SAVE = 'cache_data/hash_train'
HASH_TRAIN_FEATURE = 'cache_data/hash_train_features.npy'
HASH_TRAIN_LABLE = 'cache_data/hash_train_lables.npy'

HASH_TEST_SAVE = 'cache_data/hash_test'
HASH_TEST_FEATURE = 'cache_data/hash_test_features.npy'
HASH_TEST_LABLE = 'cache_data/hash_test_lables.npy'
FEATURE_NAME = ['site_id', 'C20', 'C19', 'site_domain', 'device_type', 'C17', \
                'device_ip', 'C14', 'C16', 'C15', 'device_conn_type', 'C1', \
                'app_category', 'site_category', 'app_domain', 'C21', 'banner_pos',\
                'app_id', 'device_id', 'hour', 'device_model', 'C18']

COL_NAME = ['C21', 'device_ip', 'site_id', 'app_id', 'C19', 'C18', \
            'device_type', 'id', 'C17', 'click', 'C15', 'C14', 'C16', \
            'device_conn_type', 'C1', 'app_category', 'site_category', \
            'app_domain', 'site_domain', 'banner_pos', 'device_id', 'C20',\
            'hour', 'device_model']

ORIGIN_TRAIN_FILE = 'origin_datas/part_train.csv'

WOE_TRAIN_FILE = 'cache_data/woe_dict_files/woe_train.csv'
WOE_TEST_FILE = 'cache_data/woe_dict_files/woe_test.csv'

WOE_TRAIN_FEATURE = 'cache_data/woe_train_features.npy'
WOE_TRAIN_LABLE = 'cache_data/woe_train_labels.npy'

WOE_TEST_FEATURE = 'cache_data/woe_test_features.npy'
WOE_TEST_LABLE = 'cache_data/woe_test_labels.npy'

HASH_FEATURE = 'cache_data/hash_features.npy'
HASH_LABLE = 'cache_data/hash_labels.npy'
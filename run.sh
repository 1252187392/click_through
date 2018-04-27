mkdir infos
#处理原始数据
echo 'read data'
time python read_data.py split > infos/read.info

#合并数据块
echo 'merge date'
time python merge_np_data.py

#LR训练
echo 'lr model'
echo 'woe'
time python Logistic/sklearn_lr.py woe > infos/woe_lr.info
echo 'hash'
time python Logistic/sklearn_lr.py hash > infos/hash_lr.info

#xgboost训练
echo 'gbdt model'
echo 'woe'
time python XGB/gbdt.py woe > infos/woe_gbdt.info
echo 'hash'
time python XGB/gbdt.py hash > infos/hash_gbdt.info

#onehot训练
echo 'onehot model'
echo 'woe'
time python XGB/one_hot.py woe > infos/woe_onehot.info
echo 'hash'
time python XGB/one_hot.py hash > infos/hash_onehot.info

#MLP
echo 'MLP'
echo 'woe'
time python MLP/mlp_model.py woe > infos/woe_MLP.txt
echo 'hash'
time python MLP/mlp_model.py hash > infos/hash_MLP.txt
echo 'woe onehot'
time python MLP/mlp_model.py woe onehot > infos/woe_onehot_MLP.txt
echo 'hash onehot'
time python MLP/mlp_model.py hash onehot > infos/hash_onehot_MLP.txt


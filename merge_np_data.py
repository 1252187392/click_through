#encoding:utf-8
import os
import numpy as np
from config import *

def merge(dirname,feature_save,lable_save):
    parts = os.listdir(dirname)
    features = None
    lables = None
    first = True
    for part in parts:
        if 'features' not in part:
            continue
        feature = np.load(dirname +'/'+ part)
        lable = np.load(dirname + '/'+part.replace('features','lables'))
        if first:
            features = feature
            lables = lable
        else:
            features = np.vstack((features, feature))
            lables = np.hstack((lables, lable))
    np.save(feature_save,features)
    np.save(lable_save,lables)

merge(WOE_TRAIN_FEATURE.replace('.npy',''),WOE_TRAIN_FEATURE,WOE_TRAIN_LABLE)
merge(HASH_TRAIN_FEATURE.replace('.npy',''),HASH_TRAIN_FEATURE,HASH_TRAIN_LABLE)
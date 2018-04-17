from xgboost import XGBClassifier
import numpy as np
import sys
sys.path.append('.')
import tensorflow as tf
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from config import *
from utils import *
from read_data import *

def data_loader(features,labels,batch_size):
    for i in range(0, features.shape[0],batch_size):
        left = min(i+batch_size,features.shape[0])
        yield features[i:left,:], labels[i:left]

mode = 'hash'
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    assert mode in ['hash','woe']
if mode == 'hash':
    train_features,test_features, train_lables, test_labels = load_hash_data()
else:
    train_features, test_features, train_lables, test_labels = load_woe_data([6, 18, 20])

input_x = tf.placeholder('float', shape = [None,166])
input_y = tf.placeholder('float', shape = [None,])
expand_y = tf.expand_dims(input_y,dim=1)
#tf.nn.batch_normalization

w1 = tf.Variable(tf.random_normal([166,32],0,1))
b1 = tf.Variable(tf.zeros([32]))
o1 = tf.matmul(input_x,w1)
logits1 = tf.matmul(input_x,w1) + b1
output_d1 = tf.nn.sigmoid(logits1)
output_d1_do = tf.nn.dropout(output_d1,keep_prob=0.7)

w2 = tf.Variable(tf.random_normal([32,1],0,1))
b2 = tf.Variable(tf.zeros([1]))
logits2 = tf.matmul(output_d1_do,w2) + b2
output_d2 = tf.nn.sigmoid(logits2)

#loss = -(input_y * tf.log(output_d1) + (1 - input_y) * tf.log(1 - output_d1))
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expand_y, logits=logits2)
target = tf.reduce_mean(loss)

train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(target)
#train_step = tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(target)
#init = tf.initialize_all_variables()
xgb = joblib.load('./models/{}_xgb.pkl'.format(mode))
encoder = joblib.load('./models/{}_one_hot_encoder.pkl'.format(mode))

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    epochs = 50
    batch_size = 512
    sess.run(init)
    for e in range(epochs):
        tot_loss = 0
        cnt = 0
        for feature,label in data_loader(train_features, train_lables, batch_size):
            leves = xgb.apply(feature)
            onehot = encoder.transform(leves).toarray()
            feature = np.hstack((feature, onehot))
            feed_dict = {input_x:feature, input_y:label}
            _, loss, o3 = sess.run([train_step,target,o1],feed_dict=feed_dict)
            tot_loss += loss
            cnt += 1
            if cnt % 10 == 0:
                saver.save(sess,save_path='./models/nn_model/batch_lr',global_step=cnt)
        tot_loss = tot_loss / cnt
        print e,tot_loss

    '''
    leves = xgb.apply(test_features)
    onehot = encoder.transform(leves).toarray()
    test_features = np.hstack((test_features, onehot))
    y_pred = sess.run(output_d2,feed_dict={input_x:test_features,input_y:test_labels})
    #print y_pred
    auc = roc_auc_score(test_labels, y_pred)
    logloss = log_loss(test_labels, y_pred)
    #print y_pred[:20]
    print 'train auc:{},log loss:{}'.format(auc, logloss)
    print sum(test_labels)
    print test_labels.shape
    print max(y_pred)
    '''
    del train_features
    del train_lables
    if mode == 'woe':
        idx, features, labels = clean_data_by_woe(ORIGIN_TEST_FILE)
        features = np.delete(features, [6, 18, 20], axis=1)
    else:
        idx, features, labels = clean_data_by_hash(ORIGIN_TEST_FILE)

    y_pred = []
    for feature, label in data_loader(features, labels, batch_size):
        leves = xgb.apply(feature)
        onehot = encoder.transform(leves).toarray()
        feature = np.hstack((feature, onehot))
        batch_pred = sess.run(output_d2, \
                          feed_dict={input_x: feature})[:,0]
        y_pred.extend(batch_pred)

    make_submit_csv(idx, y_pred, 'submit_csvs/{}_xgb_lr.csv'.format(mode))
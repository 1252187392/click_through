import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss,roc_auc_score
from sklearn.model_selection import train_test_split

def load_hash_data():
    idx = np.load('cache_data/hash_idx.npy')
    features = np.load('cache_data/hash_features.npy')
    labels = np.load('cache_data/hash_labels.npy')
    features, test_features, labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=101)

    return features, test_features, labels, test_labels

def load_woe_data():
    train_features = np.load('cache_data/woe_train_features.npy')
    train_labels = np.load('cache_data/woe_train_labels.npy')

    test_features = np.load('cache_data/woe_test_features.npy')
    test_labels = np.load('cache_data/woe_test_labels.npy')
    return np.array(train_features), np.array(test_features), np.array(train_labels), np.array(test_labels)


def data_loader(features,labels,batch_size):
    for i in range(0, features.shape[0],batch_size):
        left = min(i+batch_size,features.shape[0])
        yield features[i:left,:], labels[i:left]

input_x = tf.placeholder('float', shape = [None,19])
input_y = tf.placeholder('float', shape = [None,])

#tf.nn.batch_normalization

tf.nn.batch_normalization
w1 = tf.Variable(tf.random_normal([19,128],0,1))
b1 = tf.Variable(tf.zeros([128]))
o1 = tf.matmul(input_x,w1)
output_d1 = tf.nn.tanh(tf.matmul(input_x,w1) + b1)


w2 = tf.Variable(tf.random_normal([128,1],0,1))
b2 = tf.Variable(tf.zeros([1]))
output_d2 = tf.nn.sigmoid(tf.matmul(output_d1,w2) + b2)

'''
w3 = tf.Variable(tf.random_normal([256,128],0,1))
b3 = tf.Variable(tf.zeros([128]))
output_d3 = tf.nn.sigmoid(tf.matmul(output_d2, w3) + b3)

w4 = tf.Variable(tf.random_normal([128, 1], 0,1))
b4 = tf.Variable(tf.zeros([1]))
output_d4 = tf.nn.sigmoid(tf.matmul(output_d3, w4) + b4)
'''

loss = -(input_y * tf.log(output_d2) + (1 - input_y) * tf.log(1 - output_d2))
target = tf.reduce_mean(loss)

train_step = tf.train.AdamOptimizer(learning_rate=0.04).minimize(target)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    features, test_features, labels, test_labels = load_woe_data()
    print features[0]
    features = np.delete(features,[6,18,20],axis=1)
    test_features = np.delete(test_features,[6,18,20],axis=1)
    print features[0]
    epochs = 30
    batch_size = 512
    sess.run(init)
    for e in range(epochs):
        tot_loss = 0
        cnt = 0
        for feature,label in data_loader(features, labels, batch_size):
            feed_dict = {input_x:feature, input_y:label}
            _, loss, o3 = sess.run([train_step,target,o1],feed_dict=feed_dict)
            tot_loss += loss
            cnt += 1
        tot_loss = tot_loss / cnt
        print e,tot_loss
    y_pred = sess.run(output_d2,feed_dict={input_x:test_features,input_y:test_labels})
    #print y_pred
    auc = roc_auc_score(test_labels, y_pred)
    logloss = log_loss(test_labels, y_pred)
    print 'train auc:{},log loss:{}'.format(auc, logloss)
    print sum(test_labels)
    print test_labels.shape
    print max(y_pred)
#encoing:utf-8
import sys
sys.path.append('.')
import tensorflow as tf
from read_data import *
from utils import *


mode = 'hash'
onehot_flag = False
if len(sys.argv) > 1:
    mode = str(sys.argv[1])
    assert mode in ['hash','woe']
if len(sys.argv) > 2:
    if str(sys.argv[2]) == 'onehot':
        onehot_flag = True

train_features,test_features, train_labels, test_labels = load_data(mode)

xgb = joblib.load('./models/{}_xgb.pkl'.format(mode))
encoder = joblib.load('./models/{}_one_hot_encoder.pkl'.format(mode))

feature_nums = train_features.shape[1]
if onehot_flag :
    leves = xgb.apply(train_features[0:1])
    feature_nums = train_features.shape[1] + encoder.transform(leves).toarray().shape[1]

print 'one hot flag:{},feature nums:{}'.format(onehot_flag,feature_nums)
print 'train shape:{},test shape:{}'.format(train_features.shape, test_features.shape)

g = tf.Graph()
with g.as_default():
    input_x = tf.placeholder('float', shape=[None, feature_nums])
    input_y = tf.placeholder('float', shape=[None, ])
    expand_y = tf.expand_dims(input_y, dim=1)

    w1 = tf.Variable(tf.random_normal([feature_nums, 128], 0, 1))
    b1 = tf.Variable(tf.zeros([128]))
    o1 = tf.matmul(input_x, w1)
    logits1 = tf.matmul(input_x, w1) + b1
    output_d1 = tf.nn.sigmoid(logits1)
    output_d1_do = tf.nn.dropout(output_d1, keep_prob=0.7)

    w2 = tf.Variable(tf.random_normal([128, 32], 0, 1))
    b2 = tf.Variable(tf.zeros([32]))
    logits2 = tf.matmul(output_d1_do, w2) + b2
    output_d2 = tf.nn.sigmoid(logits2)
    output_d2_do = tf.nn.dropout(output_d2, keep_prob=0.7)

    w3 = tf.Variable(tf.random_normal([32, 1], 0, 1))
    b3 = tf.Variable(tf.zeros([1]))
    logits3 = tf.matmul(output_d2_do, w3) + b3
    output_d3 = tf.nn.sigmoid(logits3)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expand_y, logits=logits3)
    target = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(target)

    saver = tf.train.Saver()

with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True)) as sess:
    epochs = 3
    batch_size = 1000
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(epochs):
        tot_loss = 0
        cnt = 0
        for feature,label in data_loader(train_features, train_labels, batch_size):
            if onehot_flag:
                leves = xgb.apply(feature)
                onehot = encoder.transform(leves).toarray()
                feature = np.hstack((feature, onehot))
            feed_dict = {input_x:feature, input_y:label}
            _, loss, o3 = sess.run([train_step,target,o1],feed_dict=feed_dict)
            tot_loss += loss
            cnt += 1
            if cnt % 1000 == 0:
                saver.save(sess,save_path='./models/nn_model/batch_lr',global_step=cnt)
                print e, tot_loss/cnt
        print e, tot_loss / cnt
    saver.save(sess, save_path='./models/nn_model/batch_lr', global_step=cnt)

    y_pred = []
    for feature, label in data_loader(test_features, test_labels,1000):
        if onehot_flag:
            leves = xgb.apply(feature)
            onehot = encoder.transform(leves).toarray()
            feature = np.hstack((feature, onehot))
        _pred = sess.run(output_d3,feed_dict={input_x:feature,input_y:label})[:, 0]
        #print y_pred
        y_pred.extend(_pred)

    auc, logloss = get_auc_logloss(test_labels,y_pred,'test')

    del train_features
    del train_labels

    idx, features, labels = load_pred_data(mode)

    y_pred = []

    for feature, label in data_loader(features, labels, batch_size):
        if onehot_flag:
            leves = xgb.apply(feature)
            onehot = encoder.transform(leves).toarray()
            feature = np.hstack((feature, onehot))
        batch_pred = sess.run(output_d3, \
                          feed_dict={input_x: feature})[:,0]
        y_pred.extend(batch_pred)

    make_submit_csv(idx, y_pred, 'submit_csvs/{}_{}_mlp.csv'.format(mode,onehot_flag))

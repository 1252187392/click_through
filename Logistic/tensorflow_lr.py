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

train_features,test_features, train_lables, test_labels = load_data(mode)

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

    w1 = tf.Variable(tf.random_normal([feature_nums, 1], 0, .00001))
    b1 = tf.Variable(tf.zeros([1]))
    logits1 = tf.matmul(input_x, w1) + b1
    output_d1 = tf.nn.sigmoid(logits1)

    tf.layers.batch_normalization

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expand_y, logits=logits1)
    target = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(learning_rate=.1).minimize(target)

    saver = tf.train.Saver()

with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True)) as sess:
    epochs = 20
    batch_size = 512
    save_path = './models/tf_lr/lr'
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(epochs):
        tot_loss = 0
        cnt = 0
        for feature,label in data_loader(train_features, train_lables, batch_size):
            if onehot_flag:
                leves = xgb.apply(feature)
                onehot = encoder.transform(leves).toarray()
                feature = np.hstack((feature, onehot))
            feed_dict = {input_x:feature, input_y:label}
            _, los= sess.run([train_step,target],feed_dict=feed_dict)
            tot_loss += los
            cnt += 1
            if cnt % 512 == 0:
                saver.save(sess, save_path=save_path, global_step=cnt)
                print e, tot_loss/cnt
        print e, tot_loss / cnt
    saver.save(sess, save_path=save_path, global_step=cnt)

    y_pred = []
    for feature, lable in data_loader(test_features, test_labels,1000):
        if onehot_flag:
            leves = xgb.apply(feature)
            onehot = encoder.transform(leves).toarray()
            feature = np.hstack((feature, onehot))
        _pred = sess.run(output_d1,feed_dict={input_x:feature,input_y:lable})[:, 0]
        #print y_pred
        y_pred.extend(_pred)

    auc, logloss = get_auc_logloss(test_labels,y_pred,'test')

    del train_features
    del train_lables

    idx, features, labels = load_pred_data(mode)
    y_pred = []

    for feature, label in data_loader(features, labels, batch_size):
        if onehot_flag:
            leves = xgb.apply(feature)
            onehot = encoder.transform(leves).toarray()
            feature = np.hstack((feature, onehot))
        batch_pred = sess.run(output_d1, \
                          feed_dict={input_x: feature})[:,0]
        y_pred.extend(batch_pred)

    make_submit_csv(idx, y_pred, 'submit_csvs/{}_{}_tflr.csv'.format(mode,onehot_flag))

# coding=utf-8

import os
import sys
import argparse
import numpy as np

sys.path.append('/Users/lizhiyi/Documents/projects_inhands/shield/')


from engine.feature import HTKFeatures
from list_utils import load_ndx
from list_utils import get_segs_from



def parse_args():
    desp = "usage: ndx htk_head htk_tail mdl_head mdl_tail"
    parser = argparse.ArgumentParser(description=desp)

    parser.add_argument('train_ndx', help="two colum ndx: mdl_id|seg_id")
    parser.add_argument('htk_head', help="htk saved dir")
    parser.add_argument('htk_tail', help="htk postfix")
    parser.add_argument('mdl_head', help="mdl saved dir")
    parser.add_argument('mdl_tail', help="mdl postfix")
    parser.add_argument('classifier', help="classifier name in sklearn")

    args = parser.parse_args()

    return args


def load_htk_dict(segs, htk_head, htk_tail):
    segs = list(set(segs))
    htk_dict = {}

    for seg in segs:
        htk_file = htk_head + seg + htk_tail
        if not os.path.exists(htk_file):
            print "error: not exists {0}".format(htk_file)
            continue

        s = HTKFeatures()
        try:
            s.read(htk_file)
        except:
            print "error: htk read failed {0}".format(htk_file)
            continue
        htk_dict[seg] = s

    return htk_dict


def save_ivec_dict(ivec_dict, ivec_head, ivec_tail):
   for seg in ivec_dict:
       ivec_file = ivec_head + seg + ivec_tail
       if not ivec_dict[seg].write(ivec_file):
            print "error: fail to write {0}".format(ivec_file)
            continue


def load_mdl_htk_dict(ndx_dict, htk_dict):
    mdl_htk_dict = {}
    for mdl in ndx_dict:
        segs = ndx_dict[mdl]
        htk_objs = []

        for seg in segs:
            if seg in htk_dict:
                htk_objs.append(htk_dict[seg])

        if htk_objs:
            mdl_htk_dict[mdl] = htk_objs
    return mdl_htk_dict


def feed_data_to_sklearn(mdl_htk_dict):
    nsample = 0
    ndim = 0

    for mdl in mdl_htk_dict:
        ndim = mdl_htk_dict[mdl][0].features.shape[1]

    for mdl in mdl_htk_dict:
        for h in mdl_htk_dict[mdl]:
            ns, nd = h.features.shape
            n = ns / nd
            nsample += n

    X = np.zeros((nsample, ndim, ndim))  # np.array(), nsample * ndim
    y = [0] * nsample  # np.array(), nsample
    cnt = 0
    for mdl in mdl_htk_dict:
        for h in mdl_htk_dict[mdl]:
            ns, nd = h.features.shape
            n = ns / nd
            for i in xrange(n):
                X[cnt, :, :] = h.features[i*ndim:(i+1)*ndim, :]
                y[cnt] = 1 if mdl == 'genuine' else 0
                cnt += 1
    return X, np.array(y)


def cnn_binary_classifier(X, y):

    np.random.seed(1337)  # for reproducibility

    # 1. Load data
    y = y[:, np.newaxis]
    nsample, ndim, ndim = X.shape

    print nsample

    A = np.zeros((nsample, 1, ndim+1, ndim))
    A[:, 0, 0:ndim, :] = X
    A[:, 0, ndim, :] = np.repeat(y, 90, axis=1)

    np.random.shuffle(A)

    print A.shape

    pct_50 = int(A.shape[0] * 0.7)
    X_train = A[0:pct_50, :, 0:ndim, :]
    y_train = list(A[0:pct_50, 0, ndim, :][:, 0])

    X_test = A[pct_50:, :, 0:ndim, :]
    y_test = list(A[pct_50:, 0, ndim, :][:, 0])


    X_all = A[:, :, 0:ndim, :]
    y_all = list(A[:, 0, ndim, :][:, 0])


    import keras
    from keras.models import Sequential, model_from_json
    #from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
    #from keras.optimizers import RMSprop, Adam

    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D

    num_classes = 2

    # convert class vectors to binary class matrices
    # 标签转换为独热码
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_all = keras.utils.to_categorical(y_all, num_classes)

    # 2. Define CNN model

    # batch大小，每处理128个样本进行一次梯度更新
    batch_size = 128
    epochs = 12

    # Another way to build your CNN
    model = Sequential()

    # 第一层为二维卷积层
    # 32 为filters卷积核的数目，也为输出的维度
    # kernel_size 卷积核的大小，3x3
    # 激活函数选为relu
    # 第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
    input_shape=(1, 90, 90)
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 data_format='channels_first',
                 input_shape=input_shape))

    # 再加一层卷积，64个卷积核
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 加最大值池化
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    # 加Dropout，断开神经元比例为25%
    model.add(Dropout(0.25))
    # 加Flatten，数据一维化
    model.add(Flatten())
    # 加Dense，输出128维
    model.add(Dense(128, activation='relu'))
    # 再一次Dropout
    model.add(Dropout(0.5))
    # 最后一层为Softmax，输出为10个分类的概率
    model.add(Dense(2, activation='softmax'))

    # 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # 训练模型，载入数据，verbose=1为输出进度条记录
    # validation_data为验证集
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test))

    model.fit(X_all, y_all,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    return model


def dl_mdl_train(X, y, classifier='cnn'):
    if classifier == "cnn":
        return cnn_binary_classifier(X, y)
    else:
        raise ValueError("unknown classifer {0}".format(classifier))
        pass


def save_mdl_dict(model, mdl_head, mdl_tail, classifier='cnn'):
    mdl_file = mdl_head + classifier + mdl_tail
    model.save(mdl_file)



if __name__ == '__main__':

    args = parse_args()
    train_ndx = args.train_ndx
    htk_head = args.htk_head
    htk_tail = args.htk_tail
    classifier = args.classifier
    mdl_tail = args.mdl_tail
    mdl_head = args.mdl_head

    #train_ndx = '../metadata/list/dl_ivec_train.ndx.10'
    #htk_head = '../metadata/all_feats/'
    #htk_tail = '.cqcc'
    #mdl_head = './'
    #mdl_tail = '.cnn'
    #classifier = 'cnn'

    ndx_dict = load_ndx(train_ndx)

    segs = get_segs_from(ndx_dict)
    htk_dict = load_htk_dict(segs, htk_head, htk_tail)
    mdl_htk_dict = load_mdl_htk_dict(ndx_dict, htk_dict)

    X, y = feed_data_to_sklearn(mdl_htk_dict)
    model = cnn_binary_classifier(X, y)
    save_mdl_dict(model, mdl_head, mdl_tail, classifier)

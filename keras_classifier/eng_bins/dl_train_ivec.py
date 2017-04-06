# coding=utf-8

import os
import sys
import argparse
import numpy as np

sys.path.append('/Users/lizhiyi/Documents/projects_inhands/shield/')

from engine import ivector
from list_utils import load_ndx
from list_utils import get_segs_from

def parse_args():
    desp = "usage: ndx ivec_head ivec_tail mdl_head mdl_tail"
    parser = argparse.ArgumentParser(description=desp)

    parser.add_argument('train_ndx', help="two colum ndx: mdl_id|seg_id")
    parser.add_argument('ivec_head', help="ivec saved dir")
    parser.add_argument('ivec_tail', help="ivec postfix")
    parser.add_argument('mdl_head', help="mdl saved dir")
    parser.add_argument('mdl_tail', help="mdl postfix")
    parser.add_argument('classifier', help="classifier name in sklearn")

    args = parser.parse_args()

    return args


def load_ivec_dict(segs, ivec_head, ivec_tail):
    segs = list(set(segs))
    ivec_dict = {}

    for seg in segs:
        ivec_file = ivec_head + seg + ivec_tail
        if not os.path.exists(ivec_file):
            print "error: not exists {0}".format(ivec_file)
            continue

        s = ivector.IVector()
        if not s.read(ivec_file):
            print "error: ivec read failed {0}".format(ivec_file)
            continue
        ivec_dict[seg] = s

    return ivec_dict


def save_ivec_dict(ivec_dict, ivec_head, ivec_tail):
   for seg in ivec_dict:
       ivec_file = ivec_head + seg + ivec_tail
       if not ivec_dict[seg].write(ivec_file):
            print "error: fail to write {0}".format(ivec_file)
            continue


def load_mdl_ivec_dict(ndx_dict, ivec_dict):
    mdl_ivec_dict = {}
    for mdl in ndx_dict:
        segs = ndx_dict[mdl]
        ivec_objs = []

        for seg in segs:
            if seg in ivec_dict:
                ivec_objs.append(ivec_dict[seg])

        if ivec_objs:
            mdl_ivec_dict[mdl] = ivec_objs
    return mdl_ivec_dict

def feed_data_to_sklearn(mdl_ivec_dict):
    nsample = 0
    ndim = 0
    for mdl in mdl_ivec_dict:
        nsample += len(mdl_ivec_dict[mdl])
        ndim = mdl_ivec_dict[mdl][0].data.shape[0]

    X = np.zeros((nsample, ndim))  # np.array(), nsample * ndim
    y = [0] * nsample # np.array(), nsample
    cnt = 0
    for mdl in mdl_ivec_dict:
        for ivec in mdl_ivec_dict[mdl]:
            X[cnt, :] = ivec.data.T
            y[cnt] = 1 if mdl == 'genuine' else 0
            cnt += 1
    print X.shape
    print np.array(y).shape
    return X, np.array(y)

def dnn_binary_classifier(X, y):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import load_model

    # 1. Load data
    y = y[:, np.newaxis]
    A = np.hstack((X, y))
    np.random.shuffle(A)

    pct_50 = A.shape[0] / 2
    X_train = A[1:pct_50, 0:400]
    y_train = A[1:pct_50, 400]

    X_dev = A[pct_50:, 0:400]
    y_dev = A[pct_50:, 400]

    # 2. Define model
    model = Sequential()
    model.add(Dense(500, input_dim=400, init='uniform', activation='relu'))
    model.add(Dense(500, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    # 3. Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4. Fit model
    model.fit(X_train, y_train, epochs=8, batch_size=10)

    # 5. Evaluation model
    scores = model.evaluate(X_train, y_train)
    print("\ntrain set: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    scores = model.evaluate(X_dev, y_dev)
    print("\ndevlp set: %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.fit(X, y, epochs=8, batch_size=10)
    return model


def dl_mdl_train(X, y, classifier='dnn'):
    if classifier == "dnn":
        return dnn_binary_classifier(X, y)
    else:
        raise ValueError("unknown classifer {0}".format(classifier))
        pass


def save_mdl_dict(model, mdl_head, mdl_tail, classifier):
    mdl_file = mdl_head + classifier + mdl_tail
    model.save(mdl_file)


if __name__ == '__main__':
    args = parse_args()
    ndx_dict = load_ndx(args.train_ndx)
    segs = get_segs_from(ndx_dict)
    ivec_dict = load_ivec_dict(segs, args.ivec_head, args.ivec_tail)
    mdl_ivec_dict = load_mdl_ivec_dict(ndx_dict, ivec_dict)
    X, y = feed_data_to_sklearn(mdl_ivec_dict)
    model = dl_mdl_train(X, y, classifier=args.classifier)
    save_mdl_dict(model, args.mdl_head, args.mdl_tail, classifier=args.classifier)



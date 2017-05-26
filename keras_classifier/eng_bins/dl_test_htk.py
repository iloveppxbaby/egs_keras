# coding=utf-8

import os
import sys
import argparse
import numpy as np

sys.path.append('/Users/lizhiyi/Documents/projects_inhands/shield/')

from engine.feature import HTKFeatures
from list_utils import load_lines
from score_format_converter import save_score_map_to_mat

TOKEN = '|'
INVALID_SCORE = -1000


def parse_args():
    desp = "usage: mdl_list mdl_head mdl_tail seg_list htk_head htk_tail score_file"
    parser = argparse.ArgumentParser(description=desp)

    parser.add_argument('classifier', help="classifier name in sklearn")
    parser.add_argument('mdl_head', help="mdl saved dir")
    parser.add_argument('mdl_tail', help="mdl postfix")
    parser.add_argument('seg_list', help="seg list")
    parser.add_argument('htk_head', help="htk saved dir")
    parser.add_argument('htk_tail', help="htk postfix")
    parser.add_argument('score_file', help="score output file")
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


def load_model(classifier, mdl_head, mdl_tail):
    model_file = mdl_head + classifier + mdl_tail
    from keras.models import load_model
    return load_model(model_file)


def feed_seg_data_to_sklearn(seg_htk_dict):
    nsample = 0
    ndim = 0

    seg_index = {}
    for seg in seg_htk_dict:
        ndim = seg_htk_dict[seg].features.shape[1]

    for seg in seg_htk_dict:
        ns, nd = seg_htk_dict[seg].features.shape
        n = ns / nd
        seg_index[seg] = range(nsample, nsample+n)
        nsample += n

    X = np.zeros((nsample, ndim, ndim))  # np.array(), nsample * ndim
    cnt = 0
    for seg in seg_htk_dict:
        ns, nd = seg_htk_dict[seg].features.shape
        n = ns / nd
        for i in xrange(n):
            X[cnt, :, :] = seg_htk_dict[seg].features[i*ndim:(i+1)*ndim, :]
            cnt += 1
    return X, seg_index


def dl_model_test(model, seg_htk_dict):

    np.random.seed(1337)  # for reproducibility

    X, seg_index = feed_seg_data_to_sklearn(seg_htk_dict)

    nsample, ndim, ndim = X.shape
    print nsample

    A = np.zeros((nsample, 1, ndim, ndim))
    A[:, 0, 0:ndim, :] = X

    score_mat = model.predict_proba(A, verbose=1)

    print score_mat.shape

    score_map = {}
    for idx, seg in enumerate(segs):
        mdl_score = {}

        if not seg_index[seg]:
            mdl_score['genuine'] = -1000
        else:
            mdl_score['genuine'] = np.mean(score_mat[seg_index[seg], 1])
        if mdl_score:
            score_map[seg] = mdl_score
    return score_map


if __name__ == '__main__':

    args = parse_args()

    classifier = args.classifier
    htk_head = args.htk_head
    htk_tail = args.htk_tail
    mdl_head = args.mdl_head
    mdl_tail = args.mdl_tail
    seg_list = args.seg_list
    score_file = args.score_file

    #seg_list = '../metadata/list/dl_ivec_test.seg.10'
    #htk_head = '../metadata/all_feats/'
    #htk_tail = '.cqcc'
    #mdl_head = './'
    #mdl_tail = '.cnn'
    #classifier = 'cnn'
    #score_file = './score.cnn'

    segs = load_lines(seg_list)
    htk_dict = load_htk_dict(segs, htk_head, htk_tail)
    model = load_model(classifier, mdl_head, mdl_tail)
    score_map = dl_model_test(model, htk_dict)
    save_score_map_to_mat(score_map, score_file, ['genuine'], segs)

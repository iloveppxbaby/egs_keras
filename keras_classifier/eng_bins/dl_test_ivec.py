
# coding=utf-8

import os
import sys
import argparse
import numpy as np

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append('/home/zhiyi/shield/')

from ivec_train import load_ivec_dict
from list_utils import load_lines
from score_format_converter import save_score_map_to_mat

TOKEN = '|'
INVALID_SCORE = -1000


def parse_args():
    desp = "usage: mdl_list mdl_head mdl_tail seg_list ivec_head ivec_tail score_file"
    parser = argparse.ArgumentParser(description=desp)

    parser.add_argument('classifier', help="classifier name in sklearn")
    parser.add_argument('mdl_head', help="mdl saved dir")
    parser.add_argument('mdl_tail', help="mdl postfix")
    parser.add_argument('seg_list', help="seg list")
    parser.add_argument('ivec_head', help="ivec saved dir")
    parser.add_argument('ivec_tail', help="ivec postfix")
    parser.add_argument('score_file', help="score output file")
    args = parser.parse_args()

    return args


def load_model(classifier, mdl_head, mdl_tail):
    model_file = mdl_head + classifier + mdl_tail
    from keras.models import load_model
    return load_model(model_file)


def dl_model_test(model, ivec_dict):

    seg_ivecs = ivec_dict.values()
    segs = ivec_dict.keys()

    nsample = len(segs)
    ndim = seg_ivecs[0].data.shape[0]

    X = np.zeros((nsample, ndim))
    cnt = 0
    for ivec in seg_ivecs:
        X[cnt, :] = ivec.data.T
        cnt += 1

    score_mat = model.predict_proba(X, batch_size=10, verbose=1)
    score_map = {}
    for idx, seg in enumerate(segs):
        mdl_score = {}
        mdl_score['genuine'] = score_mat[idx, 0]
        if mdl_score:
            score_map[seg] = mdl_score
    return score_map


if __name__ == '__main__':

    args = parse_args()
    segs = load_lines(args.seg_list)
    ivec_dict = load_ivec_dict(segs, args.ivec_head, args.ivec_tail)
    model = load_model(args.classifier, args.mdl_head, args.mdl_tail)
    score_map = dl_model_test(model, ivec_dict)
    save_score_map_to_mat(score_map, args.score_file, ['genuine'], segs)

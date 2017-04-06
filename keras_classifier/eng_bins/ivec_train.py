# coding=utf-8

import os
import sys
import argparse
import numpy as np

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append('/home/zhiyi/shield/')
sys.path.append('/Users/lizhiyi/Documents/projects_inhands/shield/')

from engine import ivector
from list_utils import load_ndx
from list_utils import get_segs_from

from engine import ivector_plda_recognizer as ipr


def parse_args():
    desp = "usage: ndx ivec_head ivec_tail mdl_head mdl_tail"
    parser = argparse.ArgumentParser(description=desp)

    parser.add_argument('train_ndx', help="two colum ndx: mdl_id|seg_id")
    parser.add_argument('ivec_head', help="ivec saved dir")
    parser.add_argument('ivec_tail', help="ivec postfix")
    parser.add_argument('mdl_head', help="mdl saved dir")
    parser.add_argument('mdl_tail', help="mdl postfix")

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


def ivec_mdl_train(mdl_ivec_dict):
    ivec_recognizer = ipr.IVectorPLDARecognizer()
    mdl_dict = ivec_recognizer.train_models(mdl_ivec_dict)
    return mdl_dict


def save_mdl_dict(mdl_dict, mdl_head, mdl_tail):
    for mdl in mdl_dict:
        mdl_file = mdl_head + mdl + mdl_tail
        if not mdl_dict[mdl].write(mdl_file):
            print "error: fail to write {0}!".format(mdl_file)
            continue


if __name__ == '__main__':

    args = parse_args()
    ndx_dict = load_ndx(args.train_ndx)
    segs = get_segs_from(ndx_dict)
    ivec_dict = load_ivec_dict(segs, args.ivec_head, args.ivec_tail)
    mdl_ivec_dict = load_mdl_ivec_dict(ndx_dict, ivec_dict)
    mdl_dict = ivec_mdl_train(mdl_ivec_dict)
    save_mdl_dict(mdl_dict, args.mdl_head, args.mdl_tail)

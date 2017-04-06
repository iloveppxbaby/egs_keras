#!/usr/bin/python

import sys
import math


def score_split(key_file, score_file):
    fi_key = open(key_file, 'rt')
    fi_score = open(score_file, 'rt')
    fo_target = open(score_file + '_target', 'wt')
    fo_nontarget = open(score_file + '_nontarget', 'wt')
    
    key = []
    for line in fi_key.readlines():
        (seg_test, seg_mdl) = line.rstrip().split(' ')
        key.append([seg_test, seg_mdl])
    #print key
    
    label = fi_score.readline().rstrip().split('|')
    len_mdl = len(label)
    for line in fi_score.readlines():
        line_split = line.rstrip().split('|')
        seg_test = line_split[0]
        for i in xrange(1, len(line_split)):
            if [seg_test, label[i]] in key:
                #print [seg_test, label[i], line_split[i]]
                fo_target.write('%s\n'%line_split[i])
            else:
                #print [seg_test, label[i]]
                fo_nontarget.write('%s\n'%line_split[i])


if __name__ == '__main__':
        if len(sys.argv) != 3:
            print("Usage 1:", sys.argv[0], "key score")
            exit()

        score_split(sys.argv[1], sys.argv[2])


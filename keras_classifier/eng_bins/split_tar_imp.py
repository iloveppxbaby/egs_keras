# coding=utf-8

import argparse

TOKEN = "|"


def load_keys(dual_key, token=TOKEN):
    keys = []
    with open(dual_key, mode='rt') as fi_key:
        for line in fi_key.readlines():
            mdl, seg = line.rstrip().split(token)
            keys.append([mdl, seg])
    return keys


def mat_score_split(keys, mat_score, token=TOKEN):
    score_tars = []
    score_imps = []
    with open(mat_score, mode='rt') as fi_score:
        label = fi_score.readline().rstrip().split('|')
        for line in fi_score.readlines():
            line_split = line.rstrip().split(token)
            seg = line_split[0]
            for i in xrange(1, len(line_split)):
                if [seg, label[i]] in keys:
                    score_tars.append(line_split[i])
                else:
                    score_imps.append(line_split[i])
    return score_tars, score_imps


def dual_score_split(keys, dual_score, token=TOKEN):
    score_tars = []
    score_imps = []
    with open(dual_score, mode='rt') as fi_score:
        for line in fi_score.readlines():
            line_split = line.split(token)
            mdl = line_split[0]
            seg = line_split[1]
            if [mdl, seg] in keys:
                score_tars.append(line_split[2])
            else:
                score_imps.append(line_split[2])
    return score_tars, score_imps


def save_score_to_file(scores, file_name):
    with open(file_name, mode='wt') as fo_tar:
        fo_tar.writelines(scores)


def parse_argument():
    desp = "usage: dual_key [--mat] mat_score"
    parser = argparse.ArgumentParser(desp)
    parser.add_argument("dual_key", help="dual_key")
    parser.add_argument("--mat", action="store_true", help="score in mat")
    parser.add_argument("score", help="score")
    parser.add_argument("tar_score", help="target score")
    parser.add_argument("imp_score", help="impostor score")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_argument()
    keys = load_keys(args.dual_key)

    if args.mat:
        score_tars, score_imps = mat_score_split(keys, args.score)
    else:
        score_tars, score_imps = dual_score_split(keys, args.score)

    save_score_to_file(score_tars, args.tar_score)
    save_score_to_file(score_imps, args.imp_score)

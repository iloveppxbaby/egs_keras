#! /usr/bin/python

from sys import exit
from sys import argv
from sklearn import metrics

def get_lines(fn):
    try:
        fs = open(fn)
        try:
            lines = [line.strip() for line in fs.readlines()]
        finally:
            fs.close()
    except IOError, (errno, strerror):
        exit("I/O ERROR (%s): %s -- %s" % (errno, strerror, fn))
    return lines


def load_data(fn):
    data = [float(d) for d in get_lines(fn)]
    fil = []
    for i in data:
        fil.append(i)
    return fil


def mean(a):
    return sum(a) / len(a)


def linspace(min, max, n):
    s = (max - min) / (n - 1)
    return [min + s * i for i in range(n)]


def geteer(t_fn, i_fn):
    t = load_data(t_fn)
    i = load_data(i_fn)
    ml = min(i) -1
    mh = max(t) + 1

    e = 1000.0
    th = 1000.0
    dcf_1 = 1000.0
    dcf_1_th = 1000.0
    dcf_2 = 1000.0
    dcf_2_th = 1000.0

    C_Miss = 1
    C_FA = 1

    P_Target_1 = 0.01
    P_NonTarget_1 = 1 - P_Target_1
    Norm_Fact_1 = (1 / C_Miss / P_Target_1)

    P_Target_2 = 0.005
    P_NonTarget_2 = 1 - P_Target_2
    Norm_Fact_2 = (1 / C_Miss / P_Target_2)

    p_fa = [0.0001, 0.0005, 0.001]  # concerned workpoint
    gap_fa = [1000] * len(p_fa)
    p_miss = [1000] * len(p_fa)


    for c in linspace(ml, mh, 1000):
        et = float(sum([tt <= c for tt in t])) / len(t)  # miss
        ei = float(sum([ii > c for ii in i])) / len(i)  # false alarm
        en = abs(et - ei)
        if e > en:
            e = en
            etm = et
            eim = ei
            th = c

        dcfn_1 = Norm_Fact_1 * ((C_Miss * et * P_Target_1) + (C_FA * ei * P_NonTarget_1))
        if (dcf_1 > dcfn_1):
            dcf_1 = dcfn_1
            dcf_1_th = c

        dcfn_2 = Norm_Fact_2 * ((C_Miss * et * P_Target_2) + (C_FA * ei * P_NonTarget_2))
        if (dcf_2 > dcfn_2):
            dcf_2 = dcfn_2
            dcf_2_th = c

        for idx, fa in enumerate(p_fa):
            g = abs(ei - fa)
            if gap_fa[idx] > g:
                gap_fa[idx] = g
                p_miss[idx] = et

    eer = (etm + eim) / 2
    mindcf16 = (dcf_1 + dcf_2) / 2

    return (eer, dcf_1, dcf_2, mindcf16, p_fa, p_miss)

def getauc(t_fn, i_fn):
    t = load_data(t_fn)
    i = load_data(i_fn)
    pred = t + i
    y = [1] * len(t) + [0] * len(i)
    fpr, tpr, thds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


if __name__ == '__main__':
    if len(argv) != 3:
        exit('usage: geteer.py <target> <non-target>')

    eer, dcf_1, dcf_2, mindcf16, p_fa, p_miss = geteer(argv[1], argv[2])
    auc = getauc(argv[1], argv[2])

    print 'eer = %.4f dcfn_1 = %.4f dcfn_2 = %.4f mindcf16 = %.4f' % (eer, dcf_1, dcf_2, mindcf16)
    print ' '.join(['fa:miss : '] + ['%.4f:%.4f' % (fa, miss) for fa, miss in zip(p_fa, p_miss)])
    print 'auc = %.4f' % (auc)


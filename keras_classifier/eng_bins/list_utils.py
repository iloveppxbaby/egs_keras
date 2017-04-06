# _*_ coding:utf-8 _*_

import os

def gen_list(tag, wave_head, list_head):
    segs = get_file_list(wave_head)
    seg_name = list_head + tag + '.seg'
    write_list_to_file(segs, seg_name)

    seg_ndx = get_seg_ndx(wave_head)
    seg_ndx_name = list_head + tag + '.segndx'
    write_list_to_file(seg_ndx, seg_ndx_name)

    ndx = get_ndx_list(wave_head)
    ndx_name = list_head + tag + '.ndx'
    write_list_to_file(ndx, ndx_name)


def load_lines(file_name):
    with open(file_name, mode='rt') as fin:
        return [x.rstrip() for x in fin.readlines()]


def load_ndx(file_name, token='|'):
    ndx_map = {}
    with open(file_name, mode='rt') as fin:
        for x in fin.readlines():
            fields = x.rstrip().split(token)
            key = fields[0]
            value = fields[1]
            if key not in ndx_map:
                ndx_map[key] = [value]
            else:
                ndx_map[key].append(value)
    return ndx_map

def get_segs_from(ndx_map):
    segs = []
    for mdl in ndx_map:
        segs += ndx_map[mdl]
    return segs

def check_dir(name):
    if not os.path.exists(name):
        os.mkdir(name)


def get_file_list(root_dir, mininum_size_filter=0, sort_on_size=True):
    files = []
    for root, dirs, fn_list in os.walk(root_dir):
        if len(fn_list) > 0:
            if sort_on_size:
                fn_list.sort(key=lambda x: os.path.getsize(os.path.join(root, x)), reverse=True)
            for fn in fn_list:
                if fn[0] == ".":
                    continue
                fullpath_name = os.path.join(root, fn)
                if os.path.getsize(fullpath_name) > mininum_size_filter:
                    base_name = os.path.basename(fullpath_name)
                    files.append(base_name)
    return files


def write_list_to_file(list_obj, output_file):
    with open(output_file, 'w') as fout:
        [fout.writelines(x+'\n') for x in list_obj]


def write_map_to_file(map_obj, output_file):
    with open(output_file, 'wt') as fout:
        for key in map_obj:
            for value in map_obj[key]:
                fout.writelines(value)


def get_seg_ndx(root_dir):
    files = get_file_list(root_dir)
    return [x+'|'+x for x in files]


def get_ndx_list(root_dir):
    files = get_file_list(root_dir)
    return [x.split('_')[0]+'|'+x for x in files]


def score_split(key_file, score_file):
    fi_key = open(key_file, 'rt')
    fi_score = open(score_file, 'rt')
    fo_target = open(score_file + '_target', 'wt')
    fo_nontarget = open(score_file + '_impostor', 'wt')

    key = []
    for line in fi_key.readlines():
        (seg_test, seg_mdl) = line.rstrip().split('|')
        key.append([seg_test, seg_mdl])

    label = fi_score.readline().rstrip().split('|')
    for line in fi_score.readlines():
        line_split = line.rstrip().split('|')
        seg_test = line_split[0]
        for i in xrange(1, len(line_split)):
            if [seg_test, label[i]] in key:
                fo_target.write('%s\n' % line_split[i])
            else:
                fo_nontarget.write('%s\n' % line_split[i])



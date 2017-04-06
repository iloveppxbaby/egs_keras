#!/usr/bin/env bash

raw_list_dir=./metadata/raw_lists/
metadata_dir=./metadata/

wav_tail=NULL

wav_head=${metadata_dir}/all_wavs/
pcm_head=${metadata_dir}/all_pcms/
fea_head=${metadata_dir}/all_feats/
mdl_head=${metadata_dir}/all_mdls/

istat_head=${metadata_dir}/all_istats/
ivec_head=${metadata_dir}/all_ivecs/
svf_head=${metadata_dir}/all_svfs/

lambda_dir=${metadata_dir}/lambdas/

list_dir=${metadata_dir}/list/
score_dir=${metadata_dir}/scores/

mkdir -p ${score_dir}
mkdir -p ${svf_head}
mkdir -p ${score_dir}
mkdir -p ${mdl_head}
mkdir -p ${fea_head}
mkdir -p ${istat_head}
mkdir -p ${lambda_dir}
mkdir -p ${list_dir}
mkdir -p ${ivec_head}

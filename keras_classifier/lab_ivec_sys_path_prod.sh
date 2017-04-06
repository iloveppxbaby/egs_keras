#!/usr/bin/env bash

. sys_path_prod.sh

mix=512

fea_tail=.cqcc.wcmvn.lab
fea_tail=.cqcc.wcmvn.pca.lab
fea_tail=.htk.lab
fea_tail=.tfc.org
fea_tail=.plp.org
fea_tail=.cqcc.lab

istat_tail=.istat${fea_tail}.${mix}
ivec_tail=.ivec${fea_tail}.${mix}

ubm_file=${lambda_dir}/lab_gsv_ubm_${mix}${fea_tail}

latent_dim=400
tv_file=${lambda_dir}/lab_ivec_tv_${mix}_${latent_dim}${fea_tail}

norm_file_dir=${lambda_dir}/lab_norm_${mix}_${latent_dim}${fea_tail}/
mkdir -p ${norm_file_dir}

F_dim=400
iter_num=20
plda_file_dir=${lambda_dir}/lab_plda_${mix}_${latent_dim}_${F_dim}_${iter_num}${fea_tail}/
mkdir -p ${plda_file_dir}


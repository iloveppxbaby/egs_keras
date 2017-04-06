#!/bin/bash

. lab_ivec_sys_path_prod.sh

# ===== train deep learning based binary ivector classifier (dl)======
classifier=dnn

genuine_mdl_name=dl_ivec_genuine
mdl_name=${genuine_mdl_name}
mdl_tail=${fea_tail}
train_ndx=${list_dir}/dl_ivec_train.ndx
>${train_ndx}
cat ${raw_list_dir}/ASVspoof2017_train.trn | grep "genuine" | cut -d" " -f1 | awk -v mdl_name=${mdl_name} '{print "genuine|"$0 }'> ${train_ndx}
cat ${raw_list_dir}/ASVspoof2017_train.trn | grep "spoof" | cut -d" " -f1 | awk -v mdl_name=${mdl_name} '{print "spoof|"$0 }' >> ${train_ndx}
#python ./eng_bins/dl_train_ivec.py ${train_ndx} ${ivec_head} ${ivec_tail} ${mdl_head} ${mdl_tail} ${classifier}


for raw_list in ${raw_list_dir}/ASVspoof2017_dev.trl # ${raw_list_dir}/ASVspoof2017_eval.trl
do

score=${score_dir}/`basename ${raw_list}`.${classifier}${mdl_tail}

seg_list=${list_dir}/dl_ivec_test.seg
cat ${raw_list} | cut -d" " -f1 > ${seg_list}
python ./eng_bins/dl_test_ivec.py ${classifier} ${mdl_head} ${mdl_tail} ${seg_list} ${ivec_head} ${ivec_tail} ${score}

awk -F"|" '{if(NR>1) print "genuine|"$1"|"$2}' ${score} > ${score}.filt
awk -F"|" '{if(NR>1) {gsub(".wav", "", $1); print $1" "$2}}' ${score} > ${score}.filt.final
done
# === get eer

key=${list_dir}/dl_ivec_asvspoof_dev.key
cat ${raw_list} | grep "genuine" | cut -d" " -f1 | awk '{print "genuine|"$0}' > ${key}

echo ${score} >> result_dl.txt
python ./eng_bins/split_tar_imp.py ${key} ${score}.filt ${score}_tar ${score}_imp
python ./eng_bins/geteer.py ${score}_tar ${score}_imp >> result_dl.txt
















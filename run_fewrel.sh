#!/bin/bash
# This script is for running fewrel classification.
#   - TASK: task name list delimited by ",". Defaults to all.
#   - DATA: data directory. Defaults to "data".
#   - MODEL: model. 
#   - LOGPATH: log directory. Defaults to "logs".
#   - SEED: random seed. Defaults to 111.

TASK=${1:-fewrel}
TASK_TYPE=${30:-classification}
DATA=${2:-/data/fewrel_shaped/}
MODEL=${3:-transformer}
SEED=${5:-42}
NEPOCHS=${9-20}
LR=${10-2e-5}
L2=${11-0}
BATCHSIZE=${16-16}
MAX_SEQ_LENGTH=${18:-512}
BERTMODEL=${19:-bert-base-uncased}

# DATA SHAPING
METAINFO=${14-subjobjposinsent}
DESCRIPTION=${20:-subjobjposinsent} 
DESCRIPTION_LEN=${41:-0}
USE_MARK=${12-True}
USE_AUG_MARK=${24:-True}
USE_MASK=${17:-False}
TYPE_MASK=${31:-False}

# options: entropy_ranked, popularity_ranked
SUBJ_TYPE_COMBO=${35:-entropy_ranked}
SUBJ_MAX_TYPES=${41:-20}
OBJ_TYPE_COMBO=${36:-entropy_ranked}
OBJ_MAX_TYPES=${41:-20}

for SEED in  101;
  do
      python models/run.py \
            --task ${TASK} \
            --task_type ${TASK_TYPE} \
            --seed ${SEED} \
            --data_dir ${DATA} \
            --log_path logs_${TASK}/run_test/ \
            --device ${GPU} \
            --model ${MODEL} \
            --optimizer adam \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --bert_model ${BERTMODEL} \
            --n_epochs ${NEPOCHS} \
            --train_split train \
            --valid_split valid test \
            --lr ${LR} \
            --l2 ${L2} \
            --using_metainfo ${METAINFO} \
            --subj_type_combo ${SUBJ_TYPE_COMBO} \
            --subj_max_types ${SUBJ_MAX_TYPES} \
            --obj_type_combo ${OBJ_TYPE_COMBO} \
            --obj_max_types ${OBJ_MAX_TYPES} \
            --use_mask ${USE_MASK} \
            --use_mark_tokens ${USE_MARK} \
            --use_type_mask ${TYPE_MASK} \
            --use_aug_mark_tokens ${USE_AUG_MARK} \
            --using_description ${DESCRIPTION} \
            --description_max_len ${DESCRIPTION_LEN} \
            --warmup_percentage 0.0 \
            --counter_unit epoch \
            --evaluation_freq 1 \
            --checkpoint_freq 1 \
            --checkpointing 1 \
            --checkpoint_metric ${TASK}/${TASK}/valid/F1:max \
            --checkpoint_task_metrics ${TASK}/${TASK}/test/F1:max,${TASK}/${TASK}/valid/F1:max \
            --batch_size ${BATCHSIZE} \
            --fix_emb 0 \
            --dataparallel 0 \
            --clear_intermediate_checkpoints 1 \
            --clear_all_checkpoints 0
done

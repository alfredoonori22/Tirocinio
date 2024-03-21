#!/bin/bash

# custom config
DATA=/work/tesi_aonori/datasets
TRAINER=ZeroshotCLIP
DATASET=fairface
CFG=$1  # rn50, rn101, vit_b32 or vit_b16
FAIRFACECLASS=$2  # select label class for FairFace

/homes/aonori/dassl_interpreter.sh /homes/aonori/Tirocinio/CoOp/train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${DATASET}/${CFG}/${FAIRFACECLASS}/ \
--eval-only \
FAIRFACECLASS ${FAIRFACECLASS}
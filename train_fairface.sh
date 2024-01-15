#!/bin/bash

# custom config
DATA=/work/tesi_aonori/CoOp_datasets
TRAINER=CoOp

DATASET=fairface
CFG=vit_b32  # config file
# CTP=$1  # class token position (end or middle)
NCTX=16  # number of context tokens
# SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
# CSC=$4  # class-specific context (False or True)
# FAIRFACECLASS=$5  # select label class for FairFace

for CTP in end  middle
do
  for SHOTS in 1 2 4 8 16
  do
    for CSC in True False
    do
      for FAIRFACECLASS in race age #gender
      do
        for SEED in 1 2 3
        do
          DIR=CoOp/output/${DATASET}/${TRAINER}/${FAIRFACECLASS}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
          if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
          else
              /homes/aonori/dassl_interpreter.sh /homes/aonori/Tirocinio/CoOp/train.py \
              --root ${DATA} \
              --seed ${SEED} \
              --trainer ${TRAINER} \
              --dataset-config-file CoOp/configs/datasets/${DATASET}.yaml \
              --config-file CoOp/configs/trainers/${TRAINER}/${CFG}.yaml \
              --output-dir ${DIR} \
              FAIRFACECLASS ${FAIRFACECLASS} \
              TRAINER.COOP.N_CTX ${NCTX} \
              TRAINER.COOP.CSC ${CSC} \
              TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
              DATASET.NUM_SHOTS ${SHOTS}
              # --fairface-class ${FAIRFACECLASS} \
              # --num-shots ${SHOTS} \ ""
          fi
        done
      done
    done
  done
done
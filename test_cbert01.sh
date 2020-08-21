#!/usr/bin/env bash

export OMP_NUM_THREADS=4

DATA_DIR=data/cwn
MODEL_DIR=model/01-fix-encoder/cbert
mkdir -p ${MODEL_DIR}

env CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --data_path ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --vocab_path ${DATA_DIR}/vocab.txt \
    --load_model ${MODEL_DIR}/model-best-***.pt \
    --batch_size 128 \
    --cuda \
    --tie_readout \
    --enc_arch cbert \
    --beam_size 2 \
    2>&1 | tee ${MODEL_DIR}/test_01.log 
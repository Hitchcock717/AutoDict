#! /bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR=data/cwn
MODEL_DIR=model/01-fix-encoder/cbert
mkdir -p ${MODEL_DIR}

python -u train.py \
    --data_path ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --vector_path ${DATA_DIR}/vec.cwn.npy \
    --vocab_path ${DATA_DIR}/vocab.txt \
    --batch_size 256 \
    --cuda \
    --enc_arch cbert \
    --fix_encoder \
    --dropout 0.2 \
    --lr 5e-4 \
    --warmup 4000 \
    --tie_readout \
    2>&1 | tee ${MODEL_DIR}/training.log


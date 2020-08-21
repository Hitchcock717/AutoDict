#! /bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR=data/oxford
MODEL_DIR=model/01-fix-encoder/enbert
mkdir -p ${MODEL_DIR}

python -u train.py \
    --data_path ${DATA_DIR} \
    --model_dir ${MODEL_DIR} \
    --vector_path ${DATA_DIR}/vec.oxford.npy \
    --vocab_path ${DATA_DIR}/vocab.txt \
    --batch_size 128 \
    --update_interval 2 \
    --cuda \
    --enc_arch enbert \
    --fix_encoder \
    --dropout 0.2 \
    --lr 5e-4 \
    --warmup 4000 \
    --tie_readout \
    2>&1 | tee ${MODEL_DIR}/training.log
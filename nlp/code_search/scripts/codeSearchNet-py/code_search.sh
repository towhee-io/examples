#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp2_ex3

make_dir $MODEL_DIR

RGPU=0,1
MODEL_NAME=code_search_clip

function train () {

echo "============TRAINING============"

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train_code_search.py \
--data_workers 5 \
--dataset_name codeSearchNet-py \
--data_dir ${DATA_DIR}/ \
--model_dir $MODEL_DIR \
--model_name $MODEL_NAME \
--pretrained train_CSN_py_1 \
--train_src train/code.original_subtoken \
--train_tgt train/doc.original \
--dev_src dev/code.original_subtoken \
--dev_tgt dev/doc.original \
--uncase True \
--use_src_word True \
--use_src_char False \
--use_tgt_word True \
--use_tgt_char False \
--max_src_len 500 \
--max_tgt_len 500 \
--emsize 512 \
--fix_embeddings False \
--src_vocab_size 65000 \
--tgt_vocab_size 30000 \
--share_decoder_embeddings True \
--max_examples -1 \
--batch_size 32 \
--test_batch_size 32 \
--num_epochs 60 \
--model_type transformer \
--num_head 8 \
--d_k 64 \
--d_v 64 \
--d_ff 2048 \
--src_pos_emb False \
--tgt_pos_emb True \
--max_relative_pos 32 \
--use_neg_dist True \
--nlayers 6 \
--trans_drop 0.2 \
--dropout_emb 0.2 \
--dropout 0.2 \
--copy_attn False \
--early_stop 5 \
--warmup_steps 0 \
--optimizer adam \
--learning_rate 0.0001 \
--lr_decay 0.99 \
--checkpoint True \
--split_decoder False
}

# train $1 $2
train

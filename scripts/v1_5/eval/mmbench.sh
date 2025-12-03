#!/bin/bash

CKPT_DIR="/path/to/checkpoint"
DATA_DIR="/path/to/dataset"

CKPT="llava-v1.5-7b"
SPLIT="mmbench_dev_20230712"

METHOD="vispruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -m llava.eval.model_vqa_mmbench \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual_token_num ${TOKEN} \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${DATA_DIR}/mmbench/${SPLIT}.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/${METHOD} \
    --experiment ${PARAM}

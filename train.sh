#!/bin/bash

TOP_DIR="." # Your path
python ${TOP_DIR}/modules/comet_dnn_train.py \
    --flagfile=run_config.txt \
    --input_list=all_files.txt

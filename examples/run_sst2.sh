#!/usr/bin/env bash

python train_astra_core.py \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --attn-alpha 1.0 \
  --ffn-alpha 1.0 \
  --tuning-mode additive \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --run-name astra_core_sst2

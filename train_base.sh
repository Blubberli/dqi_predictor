#!/bin/bash

args=(
  # data arguments
  --data_dir europolis_dqi.csv
  --max_seq_length 256
  # model arguments
#  --model_name_or_path bert-base-cased
  --model_name_or_path roberta-base
#  --model_name_or_path allenai/longformer-base-4096
#  --model_name_or_path google/bigbird-roberta-base
  # training arguments
  --gradient_accumulation_steps 1
  --output_dir ignored
  --save_total_limit 10
  --learning_rate 1e-5
  --labels_num 5
#  --warmup_ratio 0.06
#  --weight_decay 0.1
  --num_train_epochs 10
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 64
  --metric_for_best_model macro_f1
  --load_best_model_at_end True
  --logging_dir ./logs
  --logging_steps 50
  --evaluation_strategy steps
  # --eval_steps 500
  # custom training arguments
  --class_weights True
  --folds_num 5
  --output_dir_prefix 5f
  )
python transformer_for_dqi.py "${args[@]}" "$@"

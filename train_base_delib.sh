#!/bin/bash

args=(
  # data arguments
  --data_dir data
  --max_seq_length 256
  --comment_parents_num 0
  --use_post_id False
#  --split_rand_state 1
  # model arguments
#  --model_name_or_path bert-base-cased
  --model_name_or_path roberta-base
#  --model_name_or_path allenai/longformer-base-4096
#  --model_name_or_path google/bigbird-roberta-base
  # training arguments
  --gradient_accumulation_steps 1
  --output_dir ignored
  --save_total_limit 10
  --learning_rate 5e-5
#  --warmup_ratio 0.06
#  --weight_decay 0.1
  --num_train_epochs 10
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 64
  --metric_for_best_model pos_f1
  --logging_dir ./logs
  --logging_steps 500
  --evaluation_strategy epoch
  --load_best_model_at_end True
  # --eval_steps 500
  # custom training arguments
  --optimize_threshold True
  --folds_num 5
  --output_dir_prefix 5f
  )
python -m moderator_intervention_full.train "${args[@]}" "$@"

export WANDB_MODE=offline

torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path /root/autodl-tmp/FastChat/output_test/checkpoint-1  \
    --data_path /root/autodl-tmp/FastChat/playground/data/dummy_mini.json \
    --bf16 True \
    --output_dir output_lima/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no"\
    --save_strategy "epoch"\
    --save_steps 1 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"\
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --lazy_preprocess True\
    --deepspeed "./configs/default_offload_opt_param_offload.json"\
    --tf32 True &>"./vicuna7b_dummy2.log"
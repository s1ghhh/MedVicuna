export WANDB_MODE=offline

torchrun --nproc_per_node=2 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path /l/users/hongyiwa/guoheng.sun/fastchat/vicuna_data/vicuna-7b-v1.1  \
    --data_path /l/users/hongyiwa/guoheng.sun/FastChat/playground/data/dummy_mini.json \
    --bf16 True \
    --output_dir output_test_99/ \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no"\
    --save_strategy "steps"\
    --save_steps 1 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"\
    --logging_steps 1 \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --lazy_preprocess True\
    --deepspeed "./configs/default_offload_opt_param_offload.json"\
    --tf32 True &>"./vicuna7b_dummy_test.log"
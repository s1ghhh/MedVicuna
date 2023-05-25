export WANDB_MODE=offline

torchrun --nproc_per_node=7 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path /root/autodl-tmp/vicuna  \
    --data_path /root/autodl-tmp/FastChat/playground/data/dummy_mini.json \
    --bf16 True \
    --output_dir output_test_fsdp/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no"\
    --save_strategy "epoch"\
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
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True &>"./vicuna7b_dummy_fsdp.log"
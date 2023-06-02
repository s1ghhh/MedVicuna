source /home/hongyiwa/.bashrc     ######

nvidia-smi

cd /l/users/hongyiwa/guoheng.sun/MedVicuna     ######

wget https://huggingface.co/datasets/s1ghhh/MedVicuna/resolve/main/medVicuna.json

conda activate fschat

deepspeed --num_gpus 4 --num_nodes 2 --hostfile hostfile \
    --master_addr hostname1######## --master_port=20001 \
    fastchat/train/train_mem.py \
    --model_name_or_path eachadea/vicuna-13b-1.1  \
    --data_path ./medVicuna.json \
    --bf16 True \
    --output_dir output_medvicuna_13b/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --save_strategy "epoch"\
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine"\
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True\
    --deepspeed "./configs/default_offload_opt_param.json"\
    --tf32 True &>>"./medvicuna_13b.log"

#!/bin/bash                     

#SBATCH --job-name=test    
#SBATCH --partition=long    # 此处需要更改      
#SBATCH --nodes=1
#SBATCH --output=/l/users/hongyiwa/guoheng.sun/fs_vicuna_%j.out       # 此处需要更改
#SBATCH --mem=200G             
#SBATCH --gres=gpu:4            
#SBATCH --cpus-per-task=24
#SBATCH --exclusive


source /home/hongyiwa/.bashrc   # 此处需要更改或注释掉

nvidia-smi

cd /l/users/hongyiwa/guoheng.sun/FastChat  # 此处需要更改为项目根目录

conda activate fschat   # 此处需要更改

export WANDB_MODE=offline  # 避免卡住，可以手动上传

torchrun --nproc_per_node=4 --master_port=20001 fastchat/train/train_mem.py \
    --model_name_or_path eachadea/vicuna-13b-1.1  \
    --data_path playground/data/medVicuna.json \
    --bf16 True \
    --output_dir output_vicuna_13b/ \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no"\
    --save_strategy "epoch"\
    --save_steps 1 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine"\
    --logging_steps 1 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True\
    --deepspeed "./configs/default_offload_opt_param_offload.json"\
    --tf32 True &>>"./vicuna_13b.log"
#!/bin/bash                     

#SBATCH --job-name=test    
#SBATCH --partition=long  
#SBATCH --nodes=2
#SBATCH --output=/l/users/hongyiwa/guoheng.sun/medvicuna_%j.out  ######
#SBATCH --mem=200G             
#SBATCH --gres=gpu:4            
#SBATCH --cpus-per-task=12



source /home/hongyiwa/.bashrc     ######

nvidia-smi

cd /l/users/hongyiwa/guoheng.sun/MedVicuna     ######

wget https://huggingface.co/datasets/s1ghhh/MedVicuna/resolve/main/medVicuna.json

conda activate fschat

export WANDB_MODE=offline

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
​
echo Node IP: $head_node_ip
export LOGLEVEL=INFO
​
srun torchrun \
    --nnodes 2 \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:20001 \
    fastchat/train/train_mem.py \
    --model_name_or_path eachadea/vicuna-13b-1.1  \
    --data_path ./medVicuna.json \
    --bf16 True \
    --output_dir output_medvicuna_13b/ \
    --num_train_epochs 3 \
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

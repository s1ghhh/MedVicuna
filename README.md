# MedVicuna - finetune vicuna-13B

## Environment

* cuda版本
  - ```conda install cuda -c nvidia/label/cuda-11.7.0```
  - ```conda install -c conda-forge cudnn=8.4.1```
* pip
  - ```pip install -r requirements.txt``` 


* 也可以直接创建conda环境
  - 修改```./fschat.yml```中的```name```与```prefix```
  - ```conda env create -f fschat.yml```


## Slurm

  - 修改```./slurm_fschat_vicuna_13B.sh```中的必要信息
  - 运行```sbatch ./slurm_fschat_vicuna_13B.sh```提交
  - ```./slurm_fschat_vicuna_13B.sh```默认使用deepspeed zero stage 3，如果OOM，则```sbatch ./slurm_fschat_vicuna_13B_offload.sh```


## 存储需求
  * deepspeed存ckpt时会存各个gpu的优化器状态，占用空间很大，目前的设置需要600G存储
  * 如果存储空间不足，则将```/home/.conda/envs/fschat/lib/python3.10/site-packages/transformers/trainer.py```中line 2349-2352的```self.deepspeed.save_checkpoint(output_dir)```注释，避免保存优化器状态。



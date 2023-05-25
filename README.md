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
  - ```sbatch ./slurm_fschat_vicuna_13B.sh```

## !!!数据集还未处理完，老师们先试试环境
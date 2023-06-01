# MedVicuna - finetune vicuna-13B

## Environment

* python
  - ```version==3.8.0```

* cuda
  - ```conda install cuda -c nvidia/label/cuda-11.7.0```

* torch
  - ```pip install torch==1.13.1+cu117 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116```

* pip
  - ```pip install -r requirements.txt``` 

* deepspeed
  - ```set DS_BUILD_OPS=0```
  - ```pip install deepspeed```
  - https://github.com/microsoft/DeepSpeed/issues/3145

* 或者直接创建conda环境
  - 修改```./fschat.yml```中的```name```与```prefix```
  - ```conda env create -f fschat.yml```

## Slurm

  - 修改```./run_medvicuna_13b.sh```中的必要信息
  - ```sbatch ./run_medvicuna_13b.sh```提交

## 存储需求
  * deepspeed存ckpt时会存各个gpu的优化器状态，占用空间很大，目前的设置需要600G存储
  * 如果存储空间不足，则将```/home/.conda/envs/fschat/lib/python3.8/site-packages/transformers/trainer.py```中line 2349-2352的```self.deepspeed.save_checkpoint(output_dir)```注释，避免保存优化器状态。



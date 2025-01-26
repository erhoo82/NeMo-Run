#!bin/sh

MOUNTS=/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/mcore.mxfp16:/opt/megatron-lm,/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/nemo.opt_fusion:/opt/NeMo
CONTAINER=/lustre/fsw/coreai_mlperf_training/slym/nemo/containers/nemo2412_rc1.sqsh
#CONTAINER=gitlab-master.nvidia.com/slym/images/nemo/nemo2412:te_opt_fusion

#### LLAMA3 8B
#python pretrain.py --model llama3_8b --fp8 --gbs 8 --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _base
#python pretrain.py --model llama3_8b --fp8 --gbs 8 --cp 1 --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _cp1
#python pretrain.py --model llama3_8b --fp8 --gbs 128 --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _opt_fusion --optim-fusion

#### LLAMA3 70B
#python pretrain.py --model llama3_70b --nodes 8 --fp8 --gbs 128 --nsys --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _base
#python pretrain.py --model llama3_70b --nodes 8 --fp8 --gbs 128 --pp 2 --vp 10 --nsys --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _pp2
#python pretrain.py --model llama3_70b --nodes 8 --fp8 --gbs 128 --pp 8 --vp 5 --cp 1 --pp-chunksize 2097152 --time 20 --nsys --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _pp8vp5cp1__2mb_pp

#### LLAMA3 405B
python pretrain.py --model llama31_405b --nodes 72 --fp8 --gbs 252 --pp-chunksize 2097152 --time 20 --nsys --wandb --mounts ${MOUNTS} --container ${CONTAINER} --name-postfix _2mb_pp

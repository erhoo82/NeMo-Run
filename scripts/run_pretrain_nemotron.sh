#!bin/sh

MOUNTS=/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/mcore.mxfp16:/opt/megatron-lm,/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/nemo.opt_fusion:/opt/NeMo
CONTAINER=/lustre/fsw/coreai_mlperf_training/slym/nemo/containers/nemo2412_rc1.sqsh
#CONTAINER=gitlab-master.nvidia.com/slym/images/nemo/nemo2412:te_opt_fusion


##### 22B
#python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --mbs 2 --nodes 8 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_mbs2__2mb_pp"
#python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --nodes 8 --tp 1 --vp 5 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_2mb_pp__tp1__vp5"
#python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --nodes 8 --tp 1 --vp 5 --mbs 2 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_2mb_pp__mbs2__tp1__vp5"
#python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --nodes 8 --tp 4 --pp 1 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_tp4pp1"
#python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --nodes 8 --tp 4 --mbs 2 --pp 1 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_tp4pp1mbs2"
python pretrain.py --model nemotron3_22b --fp8 --wandb --gbs 256 --nodes 8 --tp 2 --mbs 1 --pp 1 --pp-chunksize 2097152 --nsys --container ${CONTAINER} --name-postfix "_tp2pp1mbs1"


##### 15B
#python pretrain.py --model nemotron4_15b --fp8 --wandb --gbs 256 --nodes 8 --nsys --container ${CONTAINER} --name-postfix "_base"
#python pretrain.py --model nemotron4_15b --fp8 --wandb --gbs 256 --tp 2 --mbs 2 --nodes 8 --nsys --container ${CONTAINER} --name-postfix "_tp2_mbs2"
#python pretrain.py --model nemotron4_15b --fp8 --wandb --gbs 256 --tp 2 --mbs 4 --nodes 8 --nsys --container ${CONTAINER} --name-postfix "_tp2_mbs4__rope_fusion"

##### 8B
#python pretrain.py --model nemotron3_8b --fp8 --wandb --gbs 256 --tp 2 --mbs 4 --nodes 8 --nsys --container ${CONTAINER} --name-postfix "_tp2_mbs4__rope_fusion"

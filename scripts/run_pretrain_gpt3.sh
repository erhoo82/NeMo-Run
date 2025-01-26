#!bin/sh

MOUNTS=/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/mcore.mxfp16:/opt/megatron-lm,/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/mounts/nemo.opt_fusion:/opt/NeMo
CONTAINER=/lustre/fsw/coreai_mlperf_training/slym/nemo/containers/nemo2412_rc1.sqsh
#CONTAINER=gitlab-master.nvidia.com/slym/images/nemo/nemo2412:te_opt_fusion

# MLPerf
python pretrain.py --model gpt_175b --fp8 --wandb --gbs 2048 --nodes 64 --nsys --container ${CONTAINER} --name-postfix "_base"
#python pretrain.py --model gpt_175b --fp8 --wandb --gbs 2048 --nodes 64 --nsys --container ${CONTAINER} --pp-chunksize 2097152 --tp-only-amax-red --name-postfix "_2mbchunk_amax"

#python pretrain.py --model gpt_175b --fp8 --wandb --nodes 64
#python pretrain.py --model gpt_175b --fp8 --wandb --mbs 1 --gbs 32 --nodes 16 --nsys --container ${CONTAINER}  --name-postfix "_base"
#python pretrain.py --model gpt_175b --fp8 --wandb --mbs 1 --gbs 32 --nodes 16 --nsys --container ${CONTAINER}  --pp-chunksize 2097152 --name-postfix "_2mbchunk_amax" --tp-only-amax-red

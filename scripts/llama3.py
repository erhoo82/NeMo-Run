import os
import argparse
from functools import partial

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.recipes.log.default import wandb_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, bf16_with_fp8_mixed
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.lightning.run import plugins
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback


container_image="/lustre/fsw/coreai_dlalgo_llm/malayn/scratch/lightning_24.09.rc3.sqsh"
job_dir="/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.09/2.0/results"
mounts=["/lustre"]
account="coreai_dlalgo_llm"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=15, help="Run time in minutes")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--model', '-m', type=str, choices=['8b', '70b', '405b'])
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--gbs', type=int, default=128)
    parser.add_argument('--mbs', type=int, default=None)
    parser.add_argument('--fp32-grad', action='store_true')
    parser.add_argument('--gc', action='store_true', help="Do the automatic garbage collection")
    parser.add_argument('--no-vboost', action='store_true', help="Disable vboost")
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--name-postfix', type=str, default="")
    return parser.parse_args()

def nsys_profile(start, end, ranks=[7]):
    return run.Config(
        NsysCallback,
        end_step=end,
        start_step=start,
        ranks=ranks,
    )

def get_llama3_tokenizer(model):
    name = "meta-llama/Llama-3.1-405B" if model == "405b" else f"meta-llama/Meta-Llama-3-{model.upper()}"
    tokenizer = run.Config(
        AutoTokenizer,
        pretrained_model_name=name,
        use_fast=True
        )
    return tokenizer

def run_pretraining(args):
    name=f"llama3_{args.model}{args.name_postfix}"

    executor = run.SlurmExecutor(
        container_image=container_image,
        container_mounts=mounts,
        time=f"0:{args.time}:00",
        account=account,
        partition="batch",
        tunnel=run.SSHTunnel(
            user="slym",
            host="login-eos",
            job_dir=job_dir,
        ),
        nodes=args.nodes,
        ntasks_per_node=8,
        mem="0",
        exclusive=True,
        #job_dir=os.path.join(job_dir, name),
    )

    if args.model == '8b':
        pretrain_recipe = llm.llama3_8b.pretrain_recipe
    elif args.model == '70b':
        pretrain_recipe = llm.llama3_70b.pretrain_recipe
    elif args.model == '405b':
        pretrain_recipe = llm.llama31_70b.pretrain_recipe
    recipe = pretrain_recipe(
            dir="/checkpoints/llama3_7b",
            name=name,
            num_nodes=args.nodes,
            num_gpus_per_node=8,
            performance_mode=True,
        )
    recipe.trainer.max_steps=100
    recipe.trainer.log_every_n_steps=1
    recipe.trainer.strategy.progress_interval=1
    recipe.trainer.check_val_every_n_epoch = None
    recipe.trainer.enable_checkpointing = False
    recipe.log.ckpt = None
    recipe.data.global_batch_size=args.gbs
    recipe.data.num_workers=2
    recipe.data.tokenizer=get_llama3_tokenizer(args.model)
    run_plugins = []

    recipe.trainer.plugins = bf16_with_fp8_mixed() if args.fp8 else bf16_mixed()
    if not args.fp32_grad:
        recipe.trainer.plugins.grad_reduce_in_fp32=False
    if args.wandb:
        recipe.log.tensorboard=None
        recipe.log.wandb=wandb_logger(project=f"llama3_{args.model}", name=name)
        executor.env_vars = {"WANDB_API_KEY": "7e0d1026de1170a68f370a390b3ab97bfdefe975"}
    if args.nsys:
        profile_ranks = [0] if recipe.trainer.strategy.pipeline_model_parallel_size == 1 else [0, args.nodes*8-1]
        run_plugins += [
            plugins.NsysPlugin(start_step=20, end_step=22, ranks=profile_ranks, nsys_trace=["nvtx", "cuda"])
        ]
    if not args.no_vboost:
        run_plugins += [plugins.PerfEnvPlugin(enable_vboost=True)]
    if not args.gc:
        recipe.trainer.callbacks.append(
            run.Config(GarbageCollectionCallback, 100, 100)
        )

    run.run(recipe, executor=executor, plugins=run_plugins, detach=True, name=name)

if __name__ == "__main__":
    run_pretraining(get_args())

import argparse
from functools import partial

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.recipes.log.default import wandb_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, bf16_with_fp8_mixed
from nemo.collections.llm.gpt.data.mock import MockDataModule
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
    parser.add_argument('--no-vboost', action='store_false', help="Disable vboost")
    parser.add_argument('--mock-dataset', action='store_true')
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
    )

    recipe = llm.llama3_8b.pretrain_recipe(
            dir="/checkpoints/llama3_7b",
            name=name,
            num_nodes=args.nodes,
            num_gpus_per_node=8,
        )
    recipe.trainer.max_steps=100
    recipe.trainer.log_every_n_steps=1
    recipe.trainer.strategy.progress_interval=1
    recipe.data.global_batch_size=args.gbs
    recipe.data.num_workers=2
    run_plugins = []

    if args.mock_dataset:
        recipe.data=run.Config(
            MockDataModule,
            tokenizer=recipe.data.tokenizer,
            micro_batch_size=recipe.data.micro_batch_size,
            global_batch_size=recipe.data.global_batch_size,
            rampup_batch_size=recipe.data.rampup_batch_size,
            num_train_samples=recipe.data.num_train_samples,
            num_val_samples=recipe.data.num_val_samples,
            num_test_samples=recipe.data.num_test_samples,
            num_workers=recipe.data.num_workers,
            persistent_workers=recipe.data.persistent_workers,
            create_attention_mask=recipe.data.create_attention_mask,
        )

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

    run.run(recipe, executor=executor, plugins=run_plugins, detach=True)

if __name__ == "__main__":
    run_pretraining(get_args())

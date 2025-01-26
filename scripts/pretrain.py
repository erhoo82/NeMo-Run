import os
import argparse
from functools import partial

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.recipes.log.default import wandb_logger
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed, bf16_with_fp8_mixed
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.lightning.run import plugins
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback, NsysCallback


job_dir="/lustre/fsw/coreai_mlperf_training/slym/nemo/nemo24.12/results"
account="coreai_dlalgo_llm"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=15, help="Run time in minutes")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--model', '-m', type=str,
                        choices=['gpt_175b', 'llama3_8b', 'llama3_70b', 'llama31_405b',
                                 'nemotron3_8b', 'nemotron4_15b', 'nemotron3_22b', 'nemotron4_340b'])
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--gbs', type=int, default=128)
    parser.add_argument('--mbs', type=int, default=None)
    parser.add_argument('--tp', type=int, default=None)
    parser.add_argument('--pp', type=int, default=None)
    parser.add_argument('--vp', type=int, default=None)
    parser.add_argument('--cp', type=int, default=None)
    parser.add_argument('--fp32-grad', action='store_true')
    parser.add_argument('--gc', action='store_true', help="Do the automatic garbage collection")
    parser.add_argument('--no-vboost', action='store_true', help="Disable vboost")
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--nsys-shape', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--name-postfix', type=str, default="")
    parser.add_argument('--mounts', type=str, default=None)
    parser.add_argument('--optim-fusion', action='store_true')
    parser.add_argument('--container', type=str, default="nvcr.io/nvidia/nemo:dev")
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--pp-chunksize', type=int, default=None)
    parser.add_argument('--tp-only-amax-red', action='store_true')
    return parser.parse_args()

def nsys_profile(start, end, ranks=[0]):
    return run.Config(
        NsysCallback,
        end_step=end,
        start_step=start,
        ranks=ranks,
        gen_shape=True,
    )

def get_llama3_tokenizer(model):
    model_size = model.split('_')[1]
    name = "meta-llama/Llama-3.1-405B" if model_size == "405b" else f"meta-llama/Meta-Llama-3-{model_size.upper()}"
    tokenizer = run.Config(
        AutoTokenizer,
        pretrained_model_name=name,
        use_fast=True
        )
    return tokenizer

def get_gpt_tokenizer():
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    data_dir = "/lustre/fsw/coreai_dlalgo_ci/datasets/the_pile/train"
    tokenizer = run.Config(
        get_nmt_tokenizer,
        library="megatron",
        model_name = "GPT2BPETokenizer",
        vocab_file = f"{data_dir}/bpe/vocab.json",
        merges_file = f"{data_dir}/bpe/merges.txt",
        )
    return tokenizer

def get_nemotron_tokenizer():
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

    data_dir="/lustre/fsw/coreai_dlalgo_ci/datasets/the_pile/train"
    tokenizer=run.Config(
        get_nmt_tokenizer,
        library="megatron",
        model_name = "GPT2BPETokenizer",
        vocab_file = f"{data_dir}/bpe/vocab.json",
        merges_file = f"{data_dir}/bpe/merges.txt",
        )

    return tokenizer

def run_pretraining(args):
    job_name = f"{args.model}{args.name_postfix}"
    mounts = ["/lustre"]
    if args.mounts is not None:
        mounts.append(args.mounts)

    executor = run.SlurmExecutor(
        container_image=args.container,
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
        #job_dir=os.path.join(job_dir, job_name),
    )
    executor.env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_FUSED_ATTN": "1",
        "NEMORUN_HOME": os.path.join(job_dir)
    }

    # Get the pre-train recipe
    pretrain_model = getattr(llm, args.model, None)
    assert pretrain_model is not None
    pretrain_recipe = pretrain_model.pretrain_recipe

    recipe = pretrain_recipe(
            dir="/checkpoints",
            name=job_name,
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
    recipe.data.num_workers = 2
    recipe.data.num_train_samples = 1000000000 # a large number

    if 'gpt' in args.model:
        recipe.data.tokenizer = get_gpt_tokenizer()
    elif 'llama3' in args.model:
        recipe.data.tokenizer = get_llama3_tokenizer(args.model)
    elif 'nemotron' in args.model:
        recipe.data.tokenizer = get_nemotron_tokenizer()

    recipe.data.global_batch_size = args.gbs
    if args.mbs is not None:
        recipe.data.micro_batch_size = args.mbs
    if args.tp_only_amax_red:
        recipe.model.config.tp_only_amax_red = True
    if args.optim_fusion:
        recipe.optim.config.use_precision_aware_optimizer = True
    if args.tp is not None:
        recipe.trainer.strategy.tensor_model_parallel_size = args.tp
    if recipe.trainer.strategy.tensor_model_parallel_size > 1:
        recipe.trainer.strategy.sequence_parallel = True
    else:
        recipe.trainer.strategy.sequence_parallel = False
    if args.pp is not None:
        recipe.trainer.strategy.pipeline_model_parallel_size = args.pp
    if args.vp is not None:
        recipe.trainer.strategy.virtual_pipeline_model_parallel_size = args.vp
    if args.cp is not None:
        recipe.trainer.strategy.context_parallel_size = args.cp

    if 'nemotron'in args.model or 'llama3' in args.model:
        recipe.model.config.apply_rope_fusion = True

    run_plugins = []
    recipe.trainer.plugins = bf16_with_fp8_mixed() if args.fp8 else bf16_mixed()
    if not args.fp32_grad:
        recipe.trainer.plugins.grad_reduce_in_fp32=False
    if args.wandb:
        recipe.log.tensorboard=None
        recipe.log.wandb=wandb_logger(project=f"nemo2412_{args.model}", name=job_name)
        executor.env_vars["WANDB_API_KEY"] = "7e0d1026de1170a68f370a390b3ab97bfdefe975"
    if args.nsys:
        profile_ranks = [0] if recipe.trainer.strategy.pipeline_model_parallel_size == 1 else [0, args.nodes*8-1]
        #recipe.trainer.callbacks.append(nsys_profile(start=20, end=22, ranks=profile_ranks))
        #executor.get_launcher().nsys_profile = True
        #executor.get_launcher().nsys_trace = ["nvtx", "cuda"]
        run_plugins += [
            plugins.NsysPlugin(start_step=20, end_step=22, ranks=profile_ranks, nsys_trace=["nvtx", "cuda"])
        ]
    if not args.no_vboost:
        run_plugins += [plugins.PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=args.pp_chunksize)]
    if not args.gc:
        recipe.trainer.callbacks.append(
            run.Config(GarbageCollectionCallback, 100, 100)
        )

    run.run(recipe, executor=executor, plugins=run_plugins, detach=True, name=job_name)

if __name__ == "__main__":
    run_pretraining(get_args())

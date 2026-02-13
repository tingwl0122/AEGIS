#!/bin/bash
# 错误检测任务的SFT训练脚本 - 7B模型多GPU版本
# 用法: ./run_detector_sft_7b_multigpu.sh <gpu数量> <保存路径>
# 推荐用法: ./run_detector_sft_7b_multigpu.sh 7 ./detector_sft_7b_output

set -x

if [ "$#" -lt 2 ]; then
    echo "用法: $0 <nproc_per_node> <save_path>"
    echo "示例: $0 7 ./detector_sft_7b_output"
    echo "注意: 建议使用7张3090显卡训练7B模型"
    exit 1
fi

nproc_per_node=$1
save_path=$2

export HF_HOME="/data_storage/zyf/.cache/huggingface/" 

export http_proxy=http://100.68.168.184:3128/ 
export https_proxy=http://100.68.168.184:3128/ 
export HTTP_PROXY=http://100.68.168.184:3128/ 
export HTTPS_PROXY=http://100.68.168.184:3128/ 



# 数据路径
TRAIN_DATA="/data_storage/zyf/zjr/mas_l/AEGIS/verl/data/maserror/converted/train.parquet"
VAL_DATA="/data_storage/zyf/zjr/mas_l/AEGIS/verl/data/maserror/converted/val.parquet"
MODEL="Qwen/Qwen2.5-7B-Instruct"

# 创建输出目录
mkdir -p "$save_path"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=64 \
    data.max_length=8192 \
    data.truncation=right \
    optim.lr=3e-6 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    optim.warmup_steps_ratio=0.1 \
    optim.lr_scheduler=cosine \
    model.partial_pretrain=$MODEL \
    model.trust_remote_code=True \
    model.lora_rank=0\
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=True \
    model.strategy=fsdp \
    trainer.default_local_dir=$save_path \
    trainer.project_name=mas_error_attribution \
    trainer.experiment_name=qwen2.7-7b-sft \
    trainer.logger=[console,wandb] \
    trainer.total_epochs=3 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$nproc_per_node 

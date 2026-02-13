#!/bin/bash
export HF_HOME="/data_storage/zyf/.cache/huggingface/" 

export http_proxy=http://100.68.168.184:3128/ 
export https_proxy=http://100.68.168.184:3128/ 
export HTTP_PROXY=http://100.68.168.184:3128/ 
export HTTPS_PROXY=http://100.68.168.184:3128/ 



# 7b instruct - no cot

# aegis
python evaluation/local_model_vllm.py \
 --use_vllm \
 --model_name Qwen/Qwen2.5-7B-Instruct \
 --gpu_ids 0,1,2,3,4,5,6,7 \
 --max_model_len 32768 \
 --gpu_memory_utilization 0.8 \
 --batch_size 8 \
 --max_new_tokens 2048 \
 --input /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/aegis.jsonl \
 --output /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results/aegis_qwen7b.jsonl

python evaluation/get_numbers.py --results evaluation/results/aegis_qwen7b.jsonl --output evaluation/results/numbers/aegis_qwen7b_numbers.txt


# whowhen
python evaluation/local_model_vllm.py \
 --use_vllm \
 --model_name Qwen/Qwen2.5-7B-Instruct \
 --gpu_ids 0,1,2,3,4,5,6,7 \
 --max_model_len 32768 \
 --gpu_memory_utilization 0.8 \
 --batch_size 8 \
 --max_new_tokens 2048 \
 --input /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/whowhen.jsonl \
 --output /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results/whowhen_qwen7b.jsonl

python evaluation/get_numbers.py --results evaluation/results/whowhen_qwen7b.jsonl --output evaluation/results/numbers/whowhen_qwen7b_numbers.txt





 python evaluation/local_model_vllm.py \
 --use_cot \
 --use_vllm \
 --model_name Qwen/Qwen2.5-7B-Instruct \
 --gpu_ids 0,1,2,3,4,5,6,7 \
 --max_model_len 32768 \
 --gpu_memory_utilization 0.8 \
 --batch_size 8 \
 --max_new_tokens 2048 \
 --input /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/aegis.jsonl \
 --output /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results/aegis_qwen7b_cot.jsonl


# aegis
python evaluation/get_numbers.py --results evaluation/results/aegis_qwen7b_cot.jsonl --output evaluation/results/numbers/aegis_qwen7b_cot_numbers.txt





 python evaluation/local_model_vllm.py \
 --use_cot \
 --use_vllm \
 --model_name Qwen/Qwen2.5-7B-Instruct \
 --gpu_ids 0,1,2,3,4,5,6,7 \
 --max_model_len 32768 \
 --gpu_memory_utilization 0.8 \
 --batch_size 8 \
 --max_new_tokens 2048 \
 --input /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/whowhen.jsonl \
 --output /data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results/whowhen_qwen7b_cot.jsonl




# whowhen
python evaluation/get_numbers.py --results evaluation/results/whowhen_qwen7b_cot.jsonl --output evaluation/results/numbers/whowhen_qwen7b_cot_numbers.txt





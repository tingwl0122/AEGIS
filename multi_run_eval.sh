#!/usr/bin/env bash
set -euo pipefail


export HF_HOME="/data_storage/zyf/.cache/huggingface/" 

export http_proxy=http://100.68.168.184:3128/ 
export https_proxy=http://100.68.168.184:3128/ 
export HTTP_PROXY=http://100.68.168.184:3128/ 
export HTTPS_PROXY=http://100.68.168.184:3128/ 




# -----------------------------------------------------------------------------
# Multi-model evaluation runner (AEGIS + Who&When)
# Supports optional --use_cot
#
# Usage:
#   # no CoT (default)
#   bash scripts/run_eval_multi_models.sh \
#     Qwen/Qwen2.5-7B-Instruct /path/to/ckpt
#
#   # with CoT
#   USE_COT=1 bash scripts/run_eval_multi_models.sh \
#     Qwen/Qwen2.5-7B-Instruct /path/to/ckpt
# -----------------------------------------------------------------------------

# -------------------- toggles --------------------
USE_COT="${USE_COT:-0}"   # 0 = no CoT (default), 1 = use CoT

# -------------------- vLLM params ----------------
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

# -------------------- data paths -----------------
IN_AEGIS="${IN_AEGIS:-/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/aegis.jsonl}"
IN_WHOWHEN="${IN_WHOWHEN:-/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/data/whowhen.jsonl}"

OUT_DIR="${OUT_DIR:-/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results}"
NUM_DIR="${NUM_DIR:-/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/results/numbers}"

EVAL_PY="${EVAL_PY:-evaluation/local_model_vllm.py}"
NUM_PY="${NUM_PY:-evaluation/get_numbers.py}"

mkdir -p "$OUT_DIR" "$NUM_DIR"

if [[ $# -lt 1 ]]; then
  echo "ERROR: provide at least one model path"
  exit 1
fi

# -------------------- helpers --------------------
model_tag() {
  # safe filename tag from model path or HF repo id
  echo "$(basename "$1")" | tr '/:' '__'
}

cot_suffix() {
  if [[ "$USE_COT" == "1" ]]; then
    echo "cot"
  else
    echo "nocot"
  fi
}

cot_flag() {
  if [[ "$USE_COT" == "1" ]]; then
    echo "--use_cot"
  else
    echo ""
  fi
}

# -------------------- runner ---------------------
run_one_model() {
  local model="$1"
  local tag
  tag="$(model_tag "$model")"
  local cot
  cot="$(cot_suffix)"

  echo "============================================================"
  echo "[MODEL] $model"
  echo "[COT]   $cot"
  echo "============================================================"

  # ---------- AEGIS ----------
  local out_aegis="${OUT_DIR}/aegis_${tag}_${cot}.jsonl"
  local num_aegis="${NUM_DIR}/aegis_${tag}_${cot}_numbers.txt"

  echo "[RUN] AEGIS → $out_aegis"
  python "$EVAL_PY" \
    --use_vllm \
    $(cot_flag) \
    --model_name "$model" \
    --gpu_ids "$GPU_IDS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --input "$IN_AEGIS" \
    --output "$out_aegis"

  echo "[NUM] AEGIS → $num_aegis"
  python "$NUM_PY" \
    --results "$out_aegis" \
    --output "$num_aegis"

  # ---------- Who & When ----------
  local out_whowhen="${OUT_DIR}/whowhen_${tag}_${cot}.jsonl"
  local num_whowhen="${NUM_DIR}/whowhen_${tag}_${cot}_numbers.txt"

  echo "[RUN] Who&When → $out_whowhen"
  python "$EVAL_PY" \
    --use_vllm \
    $(cot_flag) \
    --model_name "$model" \
    --gpu_ids "$GPU_IDS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --input "$IN_WHOWHEN" \
    --output "$out_whowhen"

  echo "[NUM] Who&When → $num_whowhen"
  python "$NUM_PY" \
    --results "$out_whowhen" \
    --output "$num_whowhen"

  echo "[DONE] $model ($cot)"
}

for model in "$@"; do
  run_one_model "$model"
done

echo "All evaluations finished."

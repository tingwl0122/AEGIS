#!/usr/bin/env python3
"""
Qwen Anomaly Detection Script with vLLM support

Multi-GPU strategy (data parallel, default):
  Spawns one INDEPENDENT vLLM process per GPU. Each process sets its own
  CUDA_VISIBLE_DEVICES so there is NO inter-GPU communication (no NCCL, no gloo).
  This avoids the all_reduce timeout bug in vLLM's built-in DP mode.

Usage:
    # Single GPU with vLLM
    python qwen_anomaly_detection.py --input data/test.jsonl --output results.jsonl --use_vllm --gpu_ids 0

    # Data parallel across 8 GPUs (default, recommended)
    python qwen_anomaly_detection.py --input data/test.jsonl --output results.jsonl --use_vllm --gpu_ids 0,1,2,3,4,5,6,7

    # Tensor parallel across 4 GPUs (must divide num_attention_heads=28)
    python qwen_anomaly_detection.py --input data/test.jsonl --output results.jsonl --use_vllm --gpu_ids 0,1,2,3 --parallel_mode tp

    # With HuggingFace (default backend)
    python qwen_anomaly_detection.py --input data/test.jsonl --output results.jsonl

    # Custom context length and memory
    python qwen_anomaly_detection.py --input data/test.jsonl --output results.jsonl --use_vllm --gpu_ids 0,1 --max_model_len 32768 --gpu_memory_utilization 0.85
"""

import json
import argparse
import os
import re
import time
import logging
import multiprocessing
from typing import Dict, List, Optional
from pathlib import Path
import sys
import hashlib
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen_anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# Utility functions
# ============================================================

def create_sample_hash(sample: Dict) -> str:
    """Create a unique hash identifier for a sample."""
    input_data = sample.get('input', {})
    hash_data = {
        'id': sample.get('id', ''),
        'query': input_data.get('query', ''),
        'conversation_history': input_data.get('conversation_history', [])
    }
    hash_string = json.dumps(hash_data, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest()


def load_existing_results(output_file: str) -> tuple[List[Dict], set]:
    """
    Load existing result file for checkpoint resumption.

    Returns:
        tuple: (list of existing results, set of processed sample hashes)
    """
    existing_results = []
    processed_hashes = set()

    if Path(output_file).exists():
        logger.info(f"Found existing result file: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    existing_results.append(result)
                    if 'original_sample' in result:
                        sample_hash = create_sample_hash(result['original_sample'])
                        processed_hashes.add(sample_hash)
                    elif 'id' in result:
                        input_data = result.get('input', {})
                        reconstructed_sample = {
                            'id': result.get('id', ''),
                            'input': {
                                'query': input_data.get('query', ''),
                                'conversation_history': input_data.get('conversation_history', [])
                            }
                        }
                        sample_hash = create_sample_hash(reconstructed_sample)
                        processed_hashes.add(sample_hash)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in existing results: {e}")
                    continue

        logger.info(f"Loaded {len(existing_results)} existing results, processed hashes: {len(processed_hashes)}")
    else:
        logger.info("Output file does not exist, starting from scratch")

    return existing_results, processed_hashes


def filter_unprocessed_samples(samples: List[Dict], processed_hashes: set) -> List[Dict]:
    """Filter out already-processed samples based on hash set."""
    unprocessed = [s for s in samples if create_sample_hash(s) not in processed_hashes]
    logger.info(f"Total: {len(samples)}, Already processed: {len(processed_hashes)}, Remaining: {len(unprocessed)}")
    return unprocessed


def append_results(results: List[Dict], output_file: str):
    """Append results to the output file (supports checkpoint resumption)."""
    if not results:
        return
    with open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Appended {len(results)} results to: {output_file}")


def load_dataset(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load dataset from a JSONL file."""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i+1}: {e}")
    return samples


def save_results(results: List[Dict], output_file: str):
    """Save results to a file (overwrites existing content)."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"Results saved to: {output_file}")


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics from results."""
    total = len(results)
    success = sum(1 for r in results if "model_detection" in r and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    return {
        "total_samples": total,
        "successful_detections": success,
        "error_samples": errors,
        "success_rate": success / total if total > 0 else 0,
    }


def extract_conversation_text(input_data: Dict) -> str:
    """Extract full conversation text from input data."""
    query = input_data.get('query', '')
    history = input_data.get('conversation_history', [])
    text = f"QUERY:\n{query}\n\nCONVERSATION HISTORY:\n"
    for entry in history:
        text += f"Step {entry.get('step', '')} - {entry.get('agent_name', '')}:\n{entry.get('content', '')}\n\n"
    return text.strip()


SYSTEM_MESSAGE = "You are a precise analyst specialized in detecting errors in multi-agent systems. Output your analysis in valid JSON format."


def truncate_conversation_to_fit(
    conversation_text: str,
    prompt_template: str,
    tokenizer,
    max_context_len: int,
    max_new_tokens: int,
    enable_thinking: bool = False,
    model_name: str = "",
) -> str:
    """
    Truncate conversation_text so the full prompt fits within the model's context window.

    Strategy:
      1. Measure token overhead from: system message + prompt template (minus placeholder)
         + chat template special tokens + generation headroom.
      2. Compute the remaining token budget for the conversation text.
      3. If the conversation fits, return as-is.
      4. Otherwise, tokenize the conversation, truncate to budget, and decode back.

    Args:
        conversation_text: The raw conversation string to (potentially) truncate.
        prompt_template: The prompt template string containing {conversation_text}.
        tokenizer: The tokenizer instance (HF or vLLM).
        max_context_len: Maximum context length of the model (e.g., 32768).
        max_new_tokens: Tokens reserved for model generation output.
        enable_thinking: Whether thinking mode is enabled (Qwen3).
        model_name: Model name string for Qwen3 detection.

    Returns:
        The (possibly truncated) conversation text.
    """
    if max_context_len is None or max_context_len <= 0:
        # Fallback: no context limit known, just do a rough char-level truncation
        return conversation_text[:100000]

    # Available tokens for the entire input (prompt + conversation)
    available_input_tokens = max_context_len - max_new_tokens
    if available_input_tokens <= 0:
        logger.warning(f"max_context_len ({max_context_len}) <= max_new_tokens ({max_new_tokens}), cannot fit any input")
        return conversation_text[:500]

    # Measure the overhead: everything in the prompt EXCEPT the conversation text
    # Use a short placeholder to measure the "wrapper" token cost
    placeholder = "PLACEHOLDER"
    prompt_with_placeholder = prompt_template.format(conversation_text=placeholder)
    messages_with_placeholder = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt_with_placeholder},
    ]

    # Apply chat template to get the full formatted string (includes special tokens)
    try:
        if "qwen3" in model_name.lower() and enable_thinking:
            formatted = tokenizer.apply_chat_template(
                messages_with_placeholder, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
        else:
            formatted = tokenizer.apply_chat_template(
                messages_with_placeholder, tokenize=False,
                add_generation_prompt=True,
            )
        overhead_tokens = len(tokenizer.encode(formatted))
    except Exception:
        # Fallback: estimate overhead as template tokens without conversation
        overhead_text = prompt_with_placeholder + SYSTEM_MESSAGE
        overhead_tokens = len(tokenizer.encode(overhead_text))

    # Subtract placeholder tokens (they shouldn't count as overhead)
    try:
        placeholder_tokens = len(tokenizer.encode(placeholder))
    except Exception:
        placeholder_tokens = 2
    overhead_tokens = max(overhead_tokens - placeholder_tokens, 0)

    # Add a small safety margin (for tokenization edge cases, BOS/EOS, etc.)
    safety_margin = 50
    conversation_budget = available_input_tokens - overhead_tokens - safety_margin

    if conversation_budget <= 0:
        logger.warning(f"No token budget left for conversation (overhead={overhead_tokens}, "
                       f"available={available_input_tokens}). Returning minimal text.")
        return conversation_text[:200]

    # Quick check: if conversation fits, return as-is
    try:
        conversation_tokens = tokenizer.encode(conversation_text)
    except Exception:
        return conversation_text[:conversation_budget * 4]  # rough char estimate

    if len(conversation_tokens) <= conversation_budget:
        return conversation_text

    # Truncate: keep the conversation within budget
    truncated_tokens = conversation_tokens[:conversation_budget]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    logger.info(f"Truncated conversation from {len(conversation_tokens)} to {conversation_budget} tokens "
                f"(overhead={overhead_tokens}, max_ctx={max_context_len}, gen={max_new_tokens})")

    return truncated_text


def build_prompt_messages(conversation_text: str, prompt_template: str) -> List[Dict]:
    """Build the chat messages list from conversation text and prompt template."""
    prompt = prompt_template.format(conversation_text=conversation_text)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]


# ============================================================
# Response parser
# ============================================================

def parse_response(response_text: str) -> Dict:
    """
    Parse model response into JSON format.
    Enhanced parser with multiple fallback strategies.
    """
    if not response_text or not response_text.strip():
        return {"faulty_agents": []}

    clean = response_text.strip()

    # Remove tool_call tags
    if '<tool_call>' in clean:
        clean = clean.replace('<tool_call>', '').replace('</tool_call>', '').strip()

    # Fix double colons (common model error)
    clean = clean.replace('::', ':')

    # Extract JSON from markdown code blocks
    if '```json' in clean:
        m = re.search(r'```json\s*(\{.*?\})\s*```', clean, re.DOTALL)
        if m:
            clean = m.group(1)
    elif '```' in clean:
        clean = re.sub(r'```[a-z]*\s*', '', clean)
        clean = clean.replace('```', '').strip()

    # Pattern 1: Standard faulty_agents JSON object
    matches = re.findall(r'\{\s*"faulty_agents"\s*:\s*\[.*?\]\s*\}', clean, re.DOTALL)
    if matches:
        for json_str in reversed(matches):
            try:
                fixed = re.sub(r',\s*}', '}', json_str)
                fixed = re.sub(r',\s*]', ']', fixed)
                result = json.loads(fixed)
                if "faulty_agents" in result:
                    return result
            except json.JSONDecodeError:
                continue

    # Pattern 2: Extract just the array
    m = re.search(r'"faulty_agents"\s*:\s*(\[.*?\])', clean, re.DOTALL)
    if m:
        try:
            return {"faulty_agents": json.loads(m.group(1))}
        except json.JSONDecodeError:
            pass

    # Pattern 3: Entire response as JSON
    try:
        fixed = re.sub(r',\s*}', '}', clean)
        fixed = re.sub(r',\s*]', ']', fixed)
        result = json.loads(fixed)
        if "faulty_agents" in result:
            return result
    except json.JSONDecodeError:
        pass

    # Pattern 4 (last resort): Regex extraction of agent_name + error_type
    agent_matches = re.findall(r'"agent_name"\s*:\s*"([^"]+)"[^}]*"error_type"\s*:\s*"([^"]+)"', clean)
    if agent_matches:
        agents = [{"agent_name": n, "error_type": t} for n, t in agent_matches]
        return {"faulty_agents": agents}

    logger.warning(f"Parse failed. Preview: {clean[:300]}")
    return {"faulty_agents": []}


# ============================================================
# Data-parallel worker (runs in a completely isolated subprocess)
# ============================================================

def _dp_worker(
    gpu_id: str,
    worker_idx: int,
    model_name: str,
    indexed_samples: List[tuple],  # List of (original_index, sample)
    prompt_template: str,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    enable_thinking: bool,
    result_queue: multiprocessing.Queue,
):
    """
    Data-parallel worker: loads an independent vLLM engine on a single GPU.

    CRITICAL: Each worker sets CUDA_VISIBLE_DEVICES to exactly ONE GPU,
    and sets data_parallel_size=1 + tensor_parallel_size=1, so there is
    NO inter-process communication. This avoids the gloo/NCCL timeout
    that vLLM's built-in DP mode causes.
    """
    # === Isolate this worker to a single GPU ===
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Prevent vLLM from trying to use distributed backends
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("RANK", None)
    os.environ.pop("LOCAL_RANK", None)
    os.environ.pop("MASTER_ADDR", None)
    os.environ.pop("MASTER_PORT", None)

    import logging as _logging
    worker_logger = _logging.getLogger(f"dp_worker_gpu{gpu_id}")
    worker_logger.setLevel(_logging.INFO)
    handler = _logging.StreamHandler(sys.stdout)
    handler.setFormatter(_logging.Formatter(f'%(asctime)s - GPU{gpu_id} - %(levelname)s - %(message)s'))
    if not worker_logger.handlers:
        worker_logger.addHandler(handler)

    samples = [s for _, s in indexed_samples]
    indices = [i for i, _ in indexed_samples]
    worker_logger.info(f"Worker {worker_idx} starting on GPU {gpu_id} with {len(samples)} samples")

    try:
        from vllm import LLM, SamplingParams

        # Detect LoRA adapter in model path
        model_path = Path(model_name)
        if model_path.exists() and (model_path / "lora_adapter").exists():
            lora_path = str(model_path / "lora_adapter")
            base_model = "Qwen/Qwen2.5-7B-Instruct"
            worker_logger.info(f"Detected LoRA checkpoint, base model: {base_model}")
        else:
            base_model = model_name
            lora_path = None

        engine_kwargs = dict(
            model=base_model,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=lora_path is not None,
            max_lora_rank=64 if lora_path else None,
            # Explicitly disable vLLM's internal data parallelism
            data_parallel_size=1,
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len

        worker_logger.info(f"Initializing vLLM engine (gpu_util={gpu_memory_utilization}, "
                           f"max_model_len={max_model_len})")

        llm = LLM(**engine_kwargs)
        tokenizer = llm.get_tokenizer()

        lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("aegis_lora", 1, lora_path)

        sampling_params = SamplingParams(
            temperature=1.0, top_p=0.99, max_tokens=max_new_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],
        )

        # Build prompts with truncation to fit context window
        # Determine the effective max context length
        effective_max_ctx = max_model_len if max_model_len else 32768  # default for Qwen2.5-7B

        conversation_texts = [extract_conversation_text(s.get('input', {})) for s in samples]

        # Truncate each conversation to fit within context window
        truncated_texts = [
            truncate_conversation_to_fit(
                conversation_text=ct,
                prompt_template=prompt_template,
                tokenizer=tokenizer,
                max_context_len=effective_max_ctx,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                model_name=model_name,
            )
            for ct in conversation_texts
        ]

        # Build chat messages
        formatted_prompts = []
        for ct in truncated_texts:
            messages = build_prompt_messages(ct, prompt_template)
            if "qwen3" in model_name.lower() and enable_thinking:
                fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            else:
                fmt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(fmt)

        # Run inference
        worker_logger.info(f"Running inference on {len(formatted_prompts)} prompts...")
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        # Assemble results
        results = []
        for sample, orig_idx, output in zip(samples, indices, outputs):
            raw = output.outputs[0].text
            detection = parse_response(raw)
            results.append((orig_idx, {
                "id": sample.get("id"),
                "metadata": sample.get("metadata"),
                "input": sample.get("input"),
                "ground_truth": sample.get("output"),
                "model_detection": detection,
                "raw_response": raw,
            }))

        worker_logger.info(f"Worker {worker_idx} finished {len(results)} samples successfully")
        result_queue.put((worker_idx, results, None))

    except Exception as e:
        worker_logger.error(f"Worker {worker_idx} failed: {e}")
        error_results = [(idx, {
            "id": samples[i].get("id") if i < len(samples) else None,
            "error": str(e),
            "timestamp": time.time(),
        }) for i, idx in enumerate(indices)]
        result_queue.put((worker_idx, error_results, str(e)))


def run_data_parallel(samples: List[Dict], args) -> List[Dict]:
    """
    Run inference in data-parallel mode: one fully independent vLLM process per GPU.

    No NCCL, no gloo, no all_reduce — each worker is a standalone process
    that only sees one GPU via CUDA_VISIBLE_DEVICES.
    """
    gpu_list = [g.strip() for g in args.gpu_ids.split(',')]
    num_gpus = len(gpu_list)
    logger.info(f"Data-parallel mode: {len(samples)} samples across {num_gpus} independent GPU workers {gpu_list}")

    # Load prompt template once (pass to workers as string)
    if args.use_cot:
        prompt_template_path = '/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/prompt_cot.txt'
    else:
        prompt_template_path = '/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/prompt.txt'
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Split samples across GPUs (round-robin for balanced lengths)
    shards: List[List[tuple]] = [[] for _ in range(num_gpus)]
    for i, sample in enumerate(samples):
        shards[i % num_gpus].append((i, sample))

    for idx, shard in enumerate(shards):
        logger.info(f"  GPU {gpu_list[idx]}: {len(shard)} samples")

    # Launch independent worker subprocesses
    # Use 'spawn' context — required for CUDA, and ensures clean process isolation
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []

    for worker_idx, gpu_id in enumerate(gpu_list):
        if not shards[worker_idx]:
            continue

        p = ctx.Process(
            target=_dp_worker,
            args=(
                gpu_id,
                worker_idx,
                args.model_name,
                shards[worker_idx],
                prompt_template,
                args.max_new_tokens,
                args.gpu_memory_utilization,
                args.max_model_len,
                args.enable_thinking,
                result_queue,
            ),
            name=f"dp_worker_gpu{gpu_id}",
        )
        p.start()
        processes.append(p)
        logger.info(f"Launched worker on GPU {gpu_id} (PID {p.pid})")

    # Collect results from all workers
    all_indexed_results = []
    errors = []
    for _ in processes:
        worker_idx, indexed_results, error = result_queue.get(timeout=7200)  # 2h timeout
        all_indexed_results.extend(indexed_results)
        if error:
            errors.append(f"Worker {worker_idx}: {error}")
        logger.info(f"Collected {len(indexed_results)} results from worker {worker_idx}")

    # Wait for processes to exit
    for p in processes:
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"Worker {p.name} still alive, terminating")
            p.terminate()
            p.join(timeout=10)

    if errors:
        logger.warning(f"Some workers had errors: {errors}")

    # Reassemble results in original sample order
    results_by_idx = {idx: result for idx, result in all_indexed_results}
    ordered_results = []
    for i in range(len(samples)):
        if i in results_by_idx:
            ordered_results.append(results_by_idx[i])
        else:
            ordered_results.append({
                "id": samples[i].get("id"),
                "error": "Worker did not return result for this sample",
                "timestamp": time.time(),
            })

    return ordered_results


# ============================================================
# Single-GPU / Tensor-Parallel / HuggingFace detector
# ============================================================

class QwenAnomalyDetector:
    """Anomaly detection using Qwen models (single-GPU, TP, or HF mode)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        enable_thinking: bool = False,
        use_vllm: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
    ):
        """
        Initialize the Qwen model.

        Args:
            model_name: Qwen model name or local path.
            lora_adapter_path: Path to LoRA adapter (optional).
            max_new_tokens: Maximum number of new tokens to generate.
            enable_thinking: Enable thinking mode (for Qwen3 models).
            use_vllm: Whether to use vLLM for accelerated inference.
            tensor_parallel_size: Tensor parallel size for vLLM (must divide num_attention_heads).
            gpu_memory_utilization: Fraction of GPU memory to use (0-1).
            max_model_len: Maximum model context length for vLLM.
        """
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Backend: {'vLLM' if use_vllm else 'HuggingFace Transformers'}")

        # Detect LoRA adapter in checkpoint directory
        model_path = Path(model_name)
        if model_path.exists() and (model_path / "lora_adapter").exists():
            lora_adapter_path = str(model_path / "lora_adapter")
            base_model_name = "Qwen/Qwen2.5-7B-Instruct"
            logger.info(f"Detected LoRA checkpoint, using base model: {base_model_name}")
        else:
            base_model_name = model_name

        if use_vllm:
            self._init_vllm(base_model_name, lora_adapter_path, tensor_parallel_size, gpu_memory_utilization, max_model_len)
        else:
            self._init_hf(base_model_name, lora_adapter_path)

        logger.info("Model loaded successfully")

    def _init_vllm(self, model_name: str, lora_path: Optional[str], tp_size: int, gpu_util: float, max_model_len: Optional[int]):
        """Initialize vLLM engine."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm")
            raise

        engine_kwargs = dict(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            dtype="bfloat16",
            enable_lora=lora_path is not None,
            max_lora_rank=64 if lora_path else None,
            data_parallel_size=1,  # Never use vLLM's internal DP
        )
        if max_model_len is not None:
            engine_kwargs["max_model_len"] = max_model_len

        logger.info(f"vLLM config: tp_size={tp_size}, gpu_util={gpu_util}, max_model_len={max_model_len}")

        self.llm = LLM(**engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = None

        if lora_path:
            logger.info(f"Enabling LoRA adapter: {lora_path}")
            self.lora_request = LoRARequest("aegis_lora", 1, lora_path)

        self.sampling_params = SamplingParams(
            temperature=1.0, top_p=0.99, max_tokens=self.max_new_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
        logger.info("vLLM engine initialized")

    def _init_hf(self, model_name: str, lora_path: Optional[str]):
        """Initialize HuggingFace Transformers backend."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if lora_path:
            logger.info(f"Loading LoRA adapter: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_prompt_template(self, prompt_file: str) -> str:
        """Load a prompt template from file."""
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def detect_anomalies_batch(self, conversation_texts: List[str], max_retries: int = 3) -> tuple[List[Dict], List[str]]:
        """
        Batch anomaly detection.

        Returns:
            Tuple of (detection results, raw responses).
        """
        if not conversation_texts:
            return [], []

        prompt_template = self.load_prompt_template('/data_storage/zyf/zjr/mas_l/AEGIS/evaluation/prompt_cot.txt')

        # Determine effective context length
        if self.use_vllm:
            # vLLM: get from engine config
            try:
                effective_max_ctx = self.llm.llm_engine.model_config.max_model_len
            except Exception:
                effective_max_ctx = 32768
        else:
            # HF: get from model config
            try:
                effective_max_ctx = getattr(self.model.config, 'max_position_embeddings', 32768)
            except Exception:
                effective_max_ctx = 32768

        # Truncate each conversation to fit within context window
        truncated_texts = [
            truncate_conversation_to_fit(
                conversation_text=ct,
                prompt_template=prompt_template,
                tokenizer=self.tokenizer,
                max_context_len=effective_max_ctx,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=self.enable_thinking,
                model_name=self.model_name,
            )
            for ct in conversation_texts
        ]

        # Build chat messages
        batch_messages = [build_prompt_messages(ct, prompt_template) for ct in truncated_texts]

        if self.use_vllm:
            return self._detect_with_vllm(batch_messages)
        else:
            return self._detect_with_hf(batch_messages, max_retries)

    def _detect_with_vllm(self, batch_messages: List[List[Dict]]) -> tuple[List[Dict], List[str]]:
        """Run batch inference using vLLM."""
        prompts = []
        for messages in batch_messages:
            if "qwen3" in self.model_name.lower() and self.enable_thinking:
                p = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            else:
                p = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p)

        try:
            outputs = self.llm.generate(prompts, self.sampling_params, lora_request=self.lora_request)
            results, raws = [], []
            for output in outputs:
                text = output.outputs[0].text
                raws.append(text)
                results.append(parse_response(text))
            return results, raws
        except Exception as e:
            logger.error(f"vLLM inference failed: {e}")
            n = len(batch_messages)
            return [{"faulty_agents": []}] * n, ["ERROR"] * n

    def _detect_with_hf(self, batch_messages: List[List[Dict]], max_retries: int) -> tuple[List[Dict], List[str]]:
        """Run batch inference using HuggingFace Transformers."""
        for attempt in range(max_retries):
            try:
                batch_texts = []
                for messages in batch_messages:
                    if "qwen3" in self.model_name.lower() and self.enable_thinking:
                        t = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    else:
                        t = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    batch_texts.append(t)

                model_inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=65536,
                ).to(self.model.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs, max_new_tokens=self.max_new_tokens,
                        temperature=1.0, top_p=0.99, do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode only newly generated tokens
                new_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
                responses = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

                results, raws = [], []
                for resp in responses:
                    raws.append(resp)
                    results.append(parse_response(resp))

                del model_inputs, generated_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return results, raws

            except Exception as e:
                logger.error(f"HF inference failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    n = len(batch_messages)
                    return [{"faulty_agents": []}] * n, ["ERROR"] * n
                time.sleep(2)

        n = len(batch_messages)
        return [{"faulty_agents": []}] * n, ["FAILED"] * n

    def evaluate_samples_batch(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate a batch of samples for anomaly detection."""
        if not samples:
            return []
        try:
            texts = [extract_conversation_text(s.get('input', {})) for s in samples]
            detections, raws = self.detect_anomalies_batch(texts)
            results = []
            for sample, det, raw in zip(samples, detections, raws):
                results.append({
                    "id": sample.get("id"),
                    "metadata": sample.get("metadata"),
                    "input": sample.get("input"),
                    "ground_truth": sample.get("output"),
                    "model_detection": det,
                    "raw_response": raw,
                })
            return results
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            return [{"id": s.get("id"), "error": str(e), "timestamp": time.time()} for s in samples]


# ============================================================
# Main
# ============================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Anomaly detection using Qwen models")
    parser.add_argument("--input", type=str, required=True, help="Input data file path (JSONL)")
    parser.add_argument("--output", type=str, required=True, help="Output result file path (JSONL)")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference (single-GPU/TP/HF mode)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking mode (Qwen3)")

    # vLLM arguments
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for accelerated inference")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3, help="Fraction of GPU memory to use (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model context length for vLLM")

    # Multi-GPU arguments
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). "
                             "With --parallel_mode dp (default): one independent model per GPU. "
                             "With --parallel_mode tp: shards model across GPUs.")
    parser.add_argument("--parallel_mode", type=str, default="dp", choices=["dp", "tp"],
                        help="Multi-GPU strategy: "
                             "'dp' = data parallel (default, works with any GPU count, no NCCL needed), "
                             "'tp' = tensor parallel (requires num_attention_heads %% num_gpus == 0)")

    parser.add_argument("--use_cot", action="store_true")

    args = parser.parse_args()

    # Determine multi-GPU setup
    gpu_list = [g.strip() for g in args.gpu_ids.split(',')] if args.gpu_ids else []
    num_gpus = len(gpu_list)

    use_data_parallel = (
        args.use_vllm
        and num_gpus > 1
        and args.parallel_mode == "dp"
    )

    # For non-DP modes, set CUDA_VISIBLE_DEVICES now
    if not use_data_parallel:
        if args.parallel_mode == "tp" and num_gpus > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            logger.info(f"Tensor parallel mode: {num_gpus} GPUs {gpu_list}")
        elif gpu_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
            logger.info(f"Single GPU mode: GPU {gpu_list[0]}")

    # Log GPU info (skip for DP mode, workers log their own)
    if not use_data_parallel and torch.cuda.is_available():
        n = torch.cuda.device_count()
        logger.info(f"Visible GPUs: {n}")
        for i in range(n):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                        f"({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)")

    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if resuming
    processed_hashes = set()
    if args.resume:
        _, processed_hashes = load_existing_results(args.output)

    # Load dataset
    logger.info(f"Loading dataset: {args.input}")
    samples = load_dataset(args.input, args.limit)
    logger.info(f"Loaded {len(samples)} samples")

    if args.resume:
        samples = filter_unprocessed_samples(samples, processed_hashes)
        if not samples:
            logger.info("All samples have been processed!")
            return

    # ========================================
    # Path 1: Data Parallel (multi-GPU)
    # ========================================
    if use_data_parallel:
        logger.info(f"Starting data-parallel inference across {num_gpus} independent GPU workers...")
        all_results = run_data_parallel(samples, args)

        # Save all results
        if args.resume:
            append_results(all_results, args.output)
        else:
            save_results(all_results, args.output)

    # ========================================
    # Path 2: Single-GPU / TP / HuggingFace
    # ========================================
    else:
        tp_size = num_gpus if (args.parallel_mode == "tp" and num_gpus > 1) else 1

        detector = QwenAnomalyDetector(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            use_vllm=args.use_vllm,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        logger.info("Starting anomaly detection...")
        batch_size = args.batch_size
        save_interval = 20
        all_results = []
        total_batches = (len(samples) - 1) // batch_size + 1

        for batch_start in tqdm(range(0, len(samples), batch_size),
                                desc="Processing batches", total=total_batches):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_samples = samples[batch_start:batch_end]

            logger.info(f"Batch {batch_start // batch_size + 1}/{total_batches}: "
                        f"samples {batch_start + 1}-{batch_end}")

            batch_results = detector.evaluate_samples_batch(batch_samples)
            all_results.extend(batch_results)

            # Save periodically
            if len(all_results) >= save_interval or batch_end == len(samples):
                if args.resume:
                    append_results(all_results, args.output)
                else:
                    if batch_start == 0:
                        save_results(all_results, args.output)
                    else:
                        append_results(all_results, args.output)
                logger.info(f"Saved {len(all_results)} results")
                all_results = []

            if not args.use_vllm:
                time.sleep(0.5)

    logger.info("Processing complete!")

    # Final metrics
    if Path(args.output).exists():
        final_results = []
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    final_results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        metrics = calculate_metrics(final_results)
        logger.info("=" * 60)
        logger.info("Final Statistics Report:")
        logger.info(f"  Total samples:          {metrics['total_samples']}")
        logger.info(f"  Successful detections:  {metrics['successful_detections']}")
        logger.info(f"  Error samples:          {metrics['error_samples']}")
        logger.info(f"  Success rate:           {metrics['success_rate']:.2%}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
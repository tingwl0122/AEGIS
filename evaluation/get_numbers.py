#!/usr/bin/env python3
"""
Evaluation script for Multi-Agent System anomaly detection.

Computes metrics at three granularity levels:
  - Pair:  (agent_name, error_type) tuples must match exactly
  - Agent: agent_name must match (ignoring error_type)
  - Error: error_type must match (ignoring agent_name)

For each level, reports:
  - Micro-F1: aggregate TP/FP/FN across all samples, then compute F1
  - Macro-F1: compute F1 per class, then average (unweighted)

Usage:
    python evaluate.py --results results.jsonl
    python evaluate.py --results results.jsonl --output metrics.json
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All 14 error modes defined in the task
ALL_ERROR_CODES = [
    "FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5",
    "FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6",
    "FM-3.1", "FM-3.2", "FM-3.3",
]


def extract_pairs(faulty_agents: List[Dict]) -> Set[Tuple[str, str]]:
    """Extract (agent_name, error_type) pairs from a faulty_agents list."""
    pairs = set()
    for entry in faulty_agents:
        agent = str(entry.get("agent_name", "")).strip()
        error = str(entry.get("error_type", "")).strip()
        if agent and error:
            pairs.add((agent, error))
    return pairs


def extract_agents(faulty_agents: List[Dict]) -> Set[str]:
    """Extract unique agent names (ignoring error type)."""
    return {
        str(entry.get("agent_name", "")).strip()
        for entry in faulty_agents
        if str(entry.get("agent_name", "")).strip()
    }


def extract_errors(faulty_agents: List[Dict]) -> Set[str]:
    """Extract unique error types (ignoring agent name)."""
    return {
        str(entry.get("error_type", "")).strip()
        for entry in faulty_agents
        if str(entry.get("error_type", "")).strip()
    }


def compute_micro_f1(all_gt: List[Set], all_pred: List[Set]) -> Dict:
    """
    Micro-F1: aggregate TP, FP, FN across all samples, then compute P/R/F1.
    Treats the problem as a single global binary classification over all items.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gt, pred in zip(all_gt, all_pred):
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_macro_f1(all_gt: List[Set], all_pred: List[Set], all_classes: List[str] = None) -> Dict:
    """
    Macro-F1: compute F1 per class independently, then take unweighted average.

    Per the paper: "Macro-F1 computes the F1-score for each class independently and then
    takes the unweighted average. This makes Macro-F1 a crucial indicator of a model's
    ability to handle infrequent error modes and avoid bias towards common error modes."

    For each class c across all N samples:
      - TP = number of samples where c is in both gt and pred
      - FP = number of samples where c is in pred but not gt
      - FN = number of samples where c is in gt but not pred
      - Per-class P, R, F1 computed from these counts
      - Macro-F1 = unweighted mean of per-class F1 scores

    Args:
        all_gt: List of ground truth label sets, one per sample.
        all_pred: List of predicted label sets, one per sample.
        all_classes: Fixed set of classes to evaluate. If provided, computes F1 for
                     each class in this list (e.g., the 14 error codes for Error-level).
                     If None, uses the union of all classes seen in gt (not pred-only
                     classes, which would inflate the denominator).

    Returns:
        Dict with macro_f1, num_classes, and per_class breakdown.
    """
    if all_classes is None:
        # Use classes that appear in ground truth (standard for Macro-F1)
        all_classes_set = set()
        for gt in all_gt:
            all_classes_set.update(gt)
        all_classes = sorted(all_classes_set)

    per_class = {}
    for c in all_classes:
        tp = sum(1 for gt, pred in zip(all_gt, all_pred) if c in gt and c in pred)
        fp = sum(1 for gt, pred in zip(all_gt, all_pred) if c in pred and c not in gt)
        fn = sum(1 for gt, pred in zip(all_gt, all_pred) if c in gt and c not in pred)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        per_class[c] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    # Macro average: unweighted mean of per-class F1
    f1_values = [v["f1"] for v in per_class.values()]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "macro_f1": macro_f1,
        "num_classes": len(all_classes),
        "per_class": per_class,
    }


def evaluate(results: List[Dict]) -> Dict:
    """
    Run full evaluation at Pair, Agent, and Error levels.

    Args:
        results: List of result dicts, each containing 'ground_truth' and 'model_detection'.

    Returns:
        Dictionary with all metrics.
    """
    # Collect ground truth and predictions for each level
    pair_gt, pair_pred = [], []
    agent_gt, agent_pred = [], []
    error_gt, error_pred = [], []

    skipped = 0
    for result in results:
        gt_data = result.get("ground_truth", {})
        pred_data = result.get("model_detection", {})

        if gt_data is None:
            skipped += 1
            continue

        gt_agents = gt_data.get("faulty_agents", []) if isinstance(gt_data, dict) else []
        pred_agents = pred_data.get("faulty_agents", []) if isinstance(pred_data, dict) else []

        # Pair level: (agent_name, error_type)
        pair_gt.append(extract_pairs(gt_agents))
        pair_pred.append(extract_pairs(pred_agents))

        # Agent level: agent_name only
        agent_gt.append(extract_agents(gt_agents))
        agent_pred.append(extract_agents(pred_agents))

        # Error level: error_type only
        error_gt.append(extract_errors(gt_agents))
        error_pred.append(extract_errors(pred_agents))

    num_samples = len(pair_gt)
    logger.info(f"Evaluating {num_samples} samples (skipped {skipped} without ground truth)")

    # --- Pair-level metrics ---
    pair_micro = compute_micro_f1(pair_gt, pair_pred)
    # Pair Macro-F1: each unique (agent_name, error_type) seen in GT is a "class"
    pair_macro = compute_macro_f1(pair_gt, pair_pred)

    # --- Agent-level metrics ---
    agent_micro = compute_micro_f1(agent_gt, agent_pred)
    # Agent Macro-F1: each unique agent_name seen in GT is a "class"
    agent_macro = compute_macro_f1(agent_gt, agent_pred)

    # --- Error-level metrics ---
    error_micro = compute_micro_f1(error_gt, error_pred)
    # Error Macro-F1: use all 14 predefined error codes as the fixed class set
    # This ensures rare error modes are weighted equally in the average
    error_macro = compute_macro_f1(error_gt, error_pred, all_classes=ALL_ERROR_CODES)

    metrics = {
        "num_samples": num_samples,
        "pair_level": {
            "micro_f1": pair_micro["f1"],
            "micro_precision": pair_micro["precision"],
            "micro_recall": pair_micro["recall"],
            "macro_f1": pair_macro["macro_f1"],
            "num_classes": pair_macro["num_classes"],
        },
        "agent_level": {
            "micro_f1": agent_micro["f1"],
            "micro_precision": agent_micro["precision"],
            "micro_recall": agent_micro["recall"],
            "macro_f1": agent_macro["macro_f1"],
            "num_classes": agent_macro["num_classes"],
        },
        "error_level": {
            "micro_f1": error_micro["f1"],
            "micro_precision": error_micro["precision"],
            "micro_recall": error_micro["recall"],
            "macro_f1": error_macro["macro_f1"],
            "num_classes": error_macro["num_classes"],
            "per_error_code": {
                code: {
                    "f1": error_macro["per_class"].get(code, {}).get("f1", 0.0),
                    "precision": error_macro["per_class"].get(code, {}).get("precision", 0.0),
                    "recall": error_macro["per_class"].get(code, {}).get("recall", 0.0),
                    "tp": error_macro["per_class"].get(code, {}).get("tp", 0),
                    "fp": error_macro["per_class"].get(code, {}).get("fp", 0),
                    "fn": error_macro["per_class"].get(code, {}).get("fn", 0),
                }
                for code in ALL_ERROR_CODES
            },
        },
        # Raw counts for debugging
        "raw_counts": {
            "pair": {"tp": pair_micro["tp"], "fp": pair_micro["fp"], "fn": pair_micro["fn"]},
            "agent": {"tp": agent_micro["tp"], "fp": agent_micro["fp"], "fn": agent_micro["fn"]},
            "error": {"tp": error_micro["tp"], "fp": error_micro["fp"], "fn": error_micro["fn"]},
        },
    }

    return metrics


def print_metrics(metrics: Dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS  ({metrics['num_samples']} samples)")
    print("=" * 70)

    for level_name in ["pair_level", "agent_level", "error_level"]:
        level = metrics[level_name]
        label = level_name.replace("_", " ").title()
        print(f"\n  {label}:")
        print(f"    Micro-F1:    {level['micro_f1']:.4f}  (P={level['micro_precision']:.4f}, R={level['micro_recall']:.4f})")
        print(f"    Macro-F1:    {level['macro_f1']:.4f}  ({level['num_classes']} classes)")

    # Per-error breakdown
    print(f"\n  Per Error Code Breakdown:")
    print(f"    {'Code':<10} {'F1':>8} {'Prec':>8} {'Rec':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"    {'-'*54}")
    for code in ALL_ERROR_CODES:
        e = metrics["error_level"]["per_error_code"][code]
        print(f"    {code:<10} {e['f1']:>8.4f} {e['precision']:>8.4f} {e['recall']:>8.4f} "
              f"{e['tp']:>6d} {e['fp']:>6d} {e['fn']:>6d}")

    # Raw counts
    print(f"\n  Raw Counts:")
    for level_name in ["pair", "agent", "error"]:
        c = metrics["raw_counts"][level_name]
        print(f"    {level_name.capitalize():>6}: TP={c['tp']}, FP={c['fp']}, FN={c['fn']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSONL file")
    parser.add_argument("--output", type=str, default=None, help="Save metrics to JSON file (optional)")
    args = parser.parse_args()

    if not Path(args.results).exists():
        logger.error(f"Results file not found: {args.results}")
        return

    # Load results
    results = []
    with open(args.results, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(results)} results from {args.results}")

    # Filter out error results (those without model_detection)
    valid_results = [r for r in results if "model_detection" in r and "error" not in r]
    error_results = [r for r in results if "error" in r]
    logger.info(f"Valid results: {len(valid_results)}, Error results: {len(error_results)}")

    if not valid_results:
        logger.error("No valid results to evaluate!")
        return

    # Compute metrics
    metrics = evaluate(valid_results)
    print_metrics(metrics)

    # Save if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
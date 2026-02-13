#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LEVELS = ["pair_level", "agent_level", "error_level"]
F1_METRICS = ["micro_f1", "macro_f1"]


def load_json(path: Path) -> Dict[str, Any]:
    # Your *_numbers.txt is actually JSON
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_filename(name: str) -> Tuple[str, str, str]:
    """
    Parse dataset/model/cotflag from:
      aegis_Qwen2.5-7B-Instruct_cot_numbers.txt
      whowhen_global_step_100_nocot_numbers.txt
      aegis_qwen7b_numbers.txt (legacy: treat as nocot)
    Returns: (dataset, model, cotflag) where cotflag in {"cot","nocot"}.
    """
    # strip suffix
    s = name
    s = re.sub(r"_numbers\.txt$", "", s)

    # dataset is first token up to first underscore
    if "_" not in s:
        raise ValueError(f"Unexpected filename (no underscores): {name}")
    dataset, rest = s.split("_", 1)

    # cot/nocot may be present as trailing token
    cotflag = "nocot"  # default legacy
    if rest.endswith("_cot"):
        cotflag = "cot"
        rest = rest[:-4]
    elif rest.endswith("_nocot"):
        cotflag = "nocot"
        rest = rest[:-6]

    model = rest
    if not model:
        model = "unknown"
    return dataset, model, cotflag


def safe_get(d: Dict[str, Any], keys: List[str], default=np.nan):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_table(files: List[Path]) -> pd.DataFrame:
    records = []
    for p in files:
        dataset, model, cotflag = parse_filename(p.name)
        d = load_json(p)

        rec = {
            "file": str(p),
            "dataset": dataset,
            "model": model,
            "cot": cotflag,
            "num_samples": safe_get(d, ["num_samples"]),
        }
        for lvl in LEVELS:
            for m in F1_METRICS:
                rec[f"{lvl}.{m}"] = safe_get(d, [lvl, m])
            rec[f"{lvl}.num_classes"] = safe_get(d, [lvl, "num_classes"])
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    # stable ordering
    df = df.sort_values(["dataset", "cot", "model"]).reset_index(drop=True)
    return df


def plot_setting(df_setting: pd.DataFrame, title: str, outpath: Path):
    """
    Make one grouped bar plot per setting:
      x-axis: model
      bars: [pair micro, pair macro, agent micro, agent macro, error micro, error macro]
    """
    models = df_setting["model"].tolist()
    x = np.arange(len(models))

    bar_cols = [
        "pair_level.micro_f1", "pair_level.macro_f1",
        "agent_level.micro_f1", "agent_level.macro_f1",
        "error_level.micro_f1", "error_level.macro_f1",
    ]
    bar_labels = [
        "pair microF1", "pair macroF1",
        "agent microF1", "agent macroF1",
        "error microF1", "error macroF1",
    ]

    width = 0.8 / len(bar_cols)
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(models)), 5))

    # for i, (col, lab) in enumerate(zip(bar_cols, bar_labels)):
    #     vals = df_setting[col].to_numpy(dtype=float)
    #     ax.bar(x + (i - (len(bar_cols) - 1) / 2) * width, vals, width, label=lab)

    for i, (col, lab) in enumerate(zip(bar_cols, bar_labels)):
        vals = df_setting[col].to_numpy(dtype=float)

        bars = ax.bar(
            x + (i - (len(bar_cols) - 1) / 2) * width,
            vals,
            width,
            label=lab,
        )

        # --- annotate each bar with percentage ---
        for rect, v in zip(bars, vals):
            if not np.isfinite(v):
                continue

            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.01,        # slightly above bar
                f"{v * 100:.1f}",               # multiply by 100, 1 decimal
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_ylabel("F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_error_code_heatmap(
    files_in_setting: List[Path],
    labels: List[str],
    outpath: Path,
    topk: int = 40,
):
    # union error codes ranked by support (tp+fn)
    support: Dict[str, int] = {}
    per_list: List[Dict[str, Any]] = []
    for p in files_in_setting:
        d = load_json(p)
        per = safe_get(d, ["error_level", "per_error_code"], default={})
        per = per if isinstance(per, dict) else {}
        per_list.append(per)
        for code, stats in per.items():
            tp = int(safe_get(stats, ["tp"], 0))
            fn = int(safe_get(stats, ["fn"], 0))
            support[code] = support.get(code, 0) + tp + fn

    codes = [c for c, _ in sorted(support.items(), key=lambda kv: kv[1], reverse=True)]
    if topk > 0:
        codes = codes[:topk]
    if not codes:
        return

    mat = np.full((len(labels), len(codes)), np.nan, dtype=float)
    for i, per in enumerate(per_list):
        for j, code in enumerate(codes):
            stats = per.get(code, {})
            mat[i, j] = float(safe_get(stats, ["f1"], np.nan))

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(codes)), max(4, 0.5 * len(labels))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title(f"Per-error-code F1 (top {len(codes)} by support)")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels(codes, rotation=90)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing *_numbers.txt")
    ap.add_argument("--outdir", required=True, help="Output directory for figs + summary.csv")
    ap.add_argument("--heatmap", action="store_true", help="Also save per-error-code heatmaps per setting")
    ap.add_argument("--topk_error_codes", type=int, default=40, help="Top-K error codes for heatmaps (0=all)")
    args = ap.parse_args()

    indir = Path(args.dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(indir.glob("*_numbers.txt"))
    if not files:
        raise SystemExit(f"No *_numbers.txt found in {indir}")

    df = build_table(files)
    df.to_csv(outdir / "summary.csv", index=False)

    # 4 settings: dataset x cotflag
    for dataset in sorted(df["dataset"].unique()):
        for cotflag in ["nocot", "cot"]:
            df_s = df[(df["dataset"] == dataset) & (df["cot"] == cotflag)].copy()
            if df_s.empty:
                continue

            title = f"{dataset} / {cotflag}: F1 across levels"
            fig_path = outdir / f"{dataset}_{cotflag}_f1.png"
            plot_setting(df_s, title=title, outpath=fig_path)

            if args.heatmap:
                # keep same ordering as df_s
                files_s = [Path(f) for f in df_s["file"].tolist()]
                labels = df_s["model"].tolist()
                hm_path = outdir / f"{dataset}_{cotflag}_error_code_f1_heatmap.png"
                plot_error_code_heatmap(
                    files_in_setting=files_s,
                    labels=labels,
                    outpath=hm_path,
                    topk=args.topk_error_codes,
                )

    print(f"[OK] Wrote figures + summary.csv to: {outdir}")


if __name__ == "__main__":
    main()

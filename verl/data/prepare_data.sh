#!/bin/bash
# Step 1: Prepare AEGIS dataset and convert to verl parquet format
#
# Usage:
#   bash scripts/prepare_data.sh [output_dir]                          # from HuggingFace
#   bash scripts/prepare_data.sh [output_dir] --aegis-repo /path/to/AEGIS  # from local repo
#
# Output: data/aegis/{train,val,test}.parquet

set -euo pipefail

OUTPUT_DIR="${1:-data/aegis}"
shift || true  # consume output_dir positional arg if present

echo "================================================"
echo "AEGIS Data Preparation"
echo "Output: ${OUTPUT_DIR}"
echo "================================================"

python3 -m data_prep.py \
    --output_dir "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "Done. Parquet files saved to ${OUTPUT_DIR}/"
echo "Files produced:"
ls -lh "${OUTPUT_DIR}"/*.parquet 2>/dev/null || echo "  (no parquet files found)"
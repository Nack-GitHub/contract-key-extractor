#!/usr/bin/env bash
# End-to-end reproduction: data → training → evaluation → demo extraction.
# Runs on Apple Silicon (MPS) in roughly 1.5 h total; CUDA is faster.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -e ".[dev]"

# 1. data
contract-ner build-thai                # synthetic Thai train/val/test + gold set → data/processed/thai
contract-ner download                  # Kaggle "Legal Contract Dataset" (anonymous) → data/raw/kaggle
contract-ner build-cuad                # CUAD spans re-aligned into full text → data/processed/cuad

# 2. models
contract-ner train --config configs/thai_wangchanberta.yaml
contract-ner train --config configs/english_cuad.yaml

# 3. document-level evaluation → reports/*.json
contract-ner evaluate --model models/thai-wangchanberta \
    --data data/processed/thai/test_gold.jsonl data/processed/thai/test_synthetic.jsonl
contract-ner evaluate --model models/english-cuad-roberta --data data/processed/cuad/test.jsonl

# 4. demo
contract-ner extract --model models/thai-wangchanberta --file examples/thai_service_contract.txt --summary
contract-ner extract --model models/english-cuad-roberta --file examples/english_service_agreement.txt --summary

python -m pytest -q

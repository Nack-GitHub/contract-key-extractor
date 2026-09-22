#!/usr/bin/env bash
# End-to-end reproduction: data → training → evaluation → demo extraction.
# Runs on Apple Silicon (MPS) in roughly 1.5 h total; CUDA is faster.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -e ".[dev,thai,baseline]"   # the English/CUAD track alone needs only ".[dev,baseline]"

# 1. data
contract-ner build-thai                # synthetic Thai train/val/test + gold set → data/processed/thai
contract-ner download                  # Kaggle "Legal Contract Dataset" (anonymous) → data/raw/kaggle
contract-ner build-cuad                # CUAD spans re-aligned into full text → data/processed/cuad

# 2. label distribution — read this before training anything; the recipes below cite it
contract-ner profile --config configs/english_cuad.yaml --out reports/profile --stem cuad_english
contract-ner profile --config configs/thai_wangchanberta.yaml --out reports/profile --stem thai_wangchanberta --no-windows

# 3. models
contract-ner train --config configs/thai_wangchanberta.yaml
contract-ner train --config configs/english_cuad.yaml
contract-ner train --config configs/english_cuad_label_aware_augmented.yaml  # the shipped English recipe
contract-ner train-crf --config configs/english_cuad.yaml --out models/english-cuad-crf   # sequence-model baseline
# optional experiments (see docs/worklog_slot_eval_and_retrain.md and docs/worklog_label_distribution.md):
#   contract-ner train --config configs/english_cuad_label_aware.yaml   # same recipe without the augmentation
#   contract-ner train --config configs/english_cuad_merged.yaml    # START_DATE folded into AGREEMENT_DATE + date augmentation
#   contract-ner train --config configs/english_cuad_deberta.yaml   # deberta-v3-base under the merged recipe (~1.5 h)

# 4. document-level evaluation → reports/*.json
contract-ner evaluate --model models/thai-wangchanberta \
    --data data/processed/thai/test_gold.jsonl data/processed/thai/test_gold_blind.jsonl data/processed/thai/test_synthetic.jsonl
contract-ner evaluate --model models/english-cuad-roberta --data data/processed/cuad/test.jsonl
contract-ner evaluate --model models/english-cuad-roberta-label-aware-augmented \
    --data data/processed/cuad/val.jsonl data/processed/cuad/test.jsonl
contract-ner evaluate --model regex --data data/processed/cuad/test.jsonl              # pattern-only baseline, same split
contract-ner evaluate --model models/english-cuad-crf --data data/processed/cuad/test.jsonl   # conditional random field baseline

# 5. demo
contract-ner extract --model models/thai-wangchanberta --file examples/thai_service_contract.txt --summary
contract-ner extract --model models/english-cuad-roberta-label-aware-augmented --file examples/english_service_agreement.txt --summary

python -m pytest -q

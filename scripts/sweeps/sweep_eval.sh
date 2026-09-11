#!/bin/bash
# Runs every eval_*.py entry point across the relevant archs/regions.
# Requires validate results to already exist (except eval_lr.py, which
# needs no validation results -- fixed baseline hyperparameters).
set -e

REGIONS=(sw w n)
CNN_ARCHES=(a b c d lin lr mlp)
MLP_ARCHES=(a b c lin)

for region in "${REGIONS[@]}"; do
  python3 scripts/eval_rf.py --region "$region"
  python3 scripts/eval_gam.py --region "$region"
  python3 scripts/eval_lr.py --region "$region"
  for arch in "${MLP_ARCHES[@]}"; do
    python3 scripts/eval_mlp.py --arch "$arch" --region "$region"
  done
  for arch in "${CNN_ARCHES[@]}"; do
    python3 scripts/eval_cnn.py --arch "$arch" --region "$region"
  done
done

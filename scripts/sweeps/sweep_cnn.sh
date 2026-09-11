#!/bin/bash
# Runs validate_cnn.py across all 7 architectures x 3 regions = 21 combinations,
# i.e. everything the original 21 cnn_t7*_valid_*.py scripts covered between them.
#
# NOTE: no equivalent driver script existed in the original repo (the 21
# scripts were presumably launched individually, e.g. via a SLURM array) --
# this is new, added for convenience/parity with the RF/GAM/LR sweep scripts.
set -e

ARCHES=(a b c d lin lr mlp)
REGIONS=(sw w n)

for arch in "${ARCHES[@]}"; do
  for region in "${REGIONS[@]}"; do
    echo "=========================================="
    echo "arch=$arch region=$region"
    echo "=========================================="
    python3 scripts/validate_cnn.py --arch "$arch" --region "$region"
  done
done

#!/bin/bash
# Same hyperparameter grid as the original gam_trial4_valid.sh, driving the
# new CLI-args entry point (scripts/validate_gam.py) instead of env vars.
set -e

REGIONS=(sw w n)
LAM=(1 3 10 30 60)
SPLINES=(5 8 12)

for region in "${REGIONS[@]}"; do
  for lam in "${LAM[@]}"; do
    for spl in "${SPLINES[@]}"; do
      echo "----------------------------------------"
      echo "region=$region lam=$lam splines=$spl"
      python3 scripts/validate_gam.py --region "$region" --lam "$lam" --n-splines "$spl"
    done
  done
done

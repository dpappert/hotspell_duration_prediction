#!/bin/bash
# Same REG/penalty/solver grid as the original lr_trial7_valid.sh, driving
# the new CLI-args entry point (scripts/validate_lr.py).
#
# NOTE: the original also looped over 10 FOLD_SPLIT values (10 different
# pre-computed split files) -- that axis is dropped here, since LR now uses
# the same single on-the-fly generate_member_splits(seed=0) scheme as
# RF/GAM/CNN/MLP (none of which repeated across multiple splits either).
set -e

REGIONS=(sw w n)
REG_VALUES=(0.1 1)
PEN_SOLV=(
  "l1 liblinear"
  "l2 lbfgs"
)

for region in "${REGIONS[@]}"; do
  for reg in "${REG_VALUES[@]}"; do
    for ps in "${PEN_SOLV[@]}"; do
      read -r pen solv <<< "$ps"
      echo "----------------------------------------"
      echo "region=$region reg=$reg pen=$pen solv=$solv"
      python3 scripts/validate_lr.py --region "$region" --reg "$reg" --penalty "$pen" --solver "$solv"
    done
  done
done

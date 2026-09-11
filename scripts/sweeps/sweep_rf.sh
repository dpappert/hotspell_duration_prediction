#!/bin/bash
# Same hyperparameter grid as the original rf_trial6_valid.sh, driving the
# new CLI-args entry point (scripts/validate_rf.py) instead of env vars.
set -e

REGIONS=(sw w n)
N_EST=(50 100 250 500)
DEPTH=(3 5 8)
MINSPLIT=(10 20)
MINLEAF=(5 10)
MAXFEAT=("log2" "0.33")
CRIT=("gini" "log_loss")

for region in "${REGIONS[@]}"; do
  for n in "${N_EST[@]}"; do
    for d in "${DEPTH[@]}"; do
      for ms in "${MINSPLIT[@]}"; do
        for ml in "${MINLEAF[@]}"; do
          for mf in "${MAXFEAT[@]}"; do
            for crit in "${CRIT[@]}"; do
              echo "----------------------------------------"
              echo "region=$region n_est=$n depth=$d msplit=$ms mleaf=$ml mfeat=$mf crit=$crit"
              python3 scripts/validate_rf.py \
                --region "$region" \
                --n-estimators "$n" \
                --max-depth "$d" \
                --min-samples-split "$ms" \
                --min-samples-leaf "$ml" \
                --max-features "$mf" \
                --criterion "$crit"
            done
          done
        done
      done
    done
  done
  echo
  echo "=================== Completed all HP runs for region=$region ==================="
  echo
done

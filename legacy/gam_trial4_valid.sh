#!/bin/bash

SCRIPT="gam_trial4_valid.py"

# Paired F,C values
FC_PAIRS=(
  "0 1"
  "1 2"
  "2 4"
)

# Other parameters
LAM=(1 3 10 30 60)
SPLINES=(5 8 12)

for pair in "${FC_PAIRS[@]}"; do
  read f c <<< "$pair"

  for lam in "${LAM[@]}"; do
    for spl in "${SPLINES[@]}"; do

      echo "----------------------------------------"
      echo "Running F=$f C=$c LAM=$lam SPLINES=$spl"

      F=$f \
      C=$c \
      LAM=$lam \
      SPLINES=$spl \
      python3 $SCRIPT

    done
  done
done
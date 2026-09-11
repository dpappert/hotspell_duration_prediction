#!/bin/bash

SCRIPT="lr_trial7_valid.py"

# Paired F,C values
FC_PAIRS=(
  "0 1"
  "1 2"
  "2 4"
)

# Outer splits
FOLD_SPLIT=(1 2 3 4 5 6 7 8 9 10)

# Other parameters
REG_VALUES=(0.1 1)

# Penalty + solver pairs
PEN_SOLV=(
  "l1 liblinear"
  "l2 lbfgs"
)

for pair in "${FC_PAIRS[@]}"; do
  read f c <<< "$pair"

  for s in "${FOLD_SPLIT[@]}"; do
    for reg in "${REG_VALUES[@]}"; do
      for ps in "${PEN_SOLV[@]}"; do

        read pen solv <<< "$ps"

        echo "----------------------------------------"
        echo "Running F=$f C=$c SPLIT=$s REG=$reg PEN=$pen SOLV=$solv"

        F=$f \
        C=$c \
        FOLD_SPLIT=$s \
        REG=$reg \
        PEN=$pen \
        SOLV=$solv \
        python3 $SCRIPT

      done
    done
  done
done

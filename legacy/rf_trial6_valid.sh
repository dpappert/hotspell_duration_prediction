#!/bin/bash

SCRIPT="rf_trial6_valid.py"

# Paired F,C values
FC_PAIRS=(
  "0 1"
  "1 2"
  "2 4"
)

# Hyperparameters
N_EST=(50 100 250 500)
DEPTH=(3 5 8)
MINSPLIT=(10 20)
MINLEAF=(5 10)
maxFEAT=("log2" "0.33")
CRIT=("gini" "log_loss")

for pair in "${FC_PAIRS[@]}"; do
  read f c <<< "$pair"

  for n in "${N_EST[@]}"; do
    for d in "${DEPTH[@]}"; do
      for ms in "${MINSPLIT[@]}"; do
        for ml in "${MINLEAF[@]}"; do
          for mf in "${maxFEAT[@]}"; do
            for crit in "${CRIT[@]}"; do

              echo "----------------------------------------"
              echo "F=$f C=$c N_EST=$n DEPTH=$d MINSPLIT=$ms MINLEAF=$ml maxFEAT=$mf CRIT=$crit"

              # Build output folder and tag exactly like Python does
              outfolder="/scratch3/dpappert/CESM2/ML/RF/scores/targets_cl${c}_HWs_m50_18512010_fmstat_1.5sd_r1/trial6/valid/"
              tag="F${f}_C${c}_nest${n}_depth${d}_msplit${ms}_mleaf${ml}_mfeat${mf}_crit${crit}"

              # Skip if file already exists
              if [ -f "${outfolder}/metrics_${tag}.csv" ]; then
                  echo "Skipping ${tag} (already exists)"
                  continue
              fi

              F=$f \
              C=$c \
              N_EST=$n \
              DEPTH=$d \
              minSPLIT=$ms \
              minLEAF=$ml \
              maxFEAT=$mf \
              CRIT=$crit \
              python3 $SCRIPT

            done
          done
        done
      done
    done
  done

  echo
  echo "=================== Completed all HP runs for F=$f C=$c ==================="
  echo
  echo
done

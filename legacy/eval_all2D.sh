#!/bin/bash

models=(
  #"LogReg2D"
  #"MLP2D"
  "CNNlin"
  #"CNNa"
  #"CNNb"
  #"CNNc"
  #"CNNd"
)

pairs=(
  "0 1"
  "1 2"
  "2 4"
)

for M in "${models[@]}"; do
    for pair in "${pairs[@]}"; do
        read -r F C <<< "$pair"

        echo "Running eval_all2D.py with M=$M, F=$F, C=$C"

        M=$M F=$F C=$C python eval_all2D.py
    done
done

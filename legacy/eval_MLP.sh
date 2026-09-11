#!/bin/bash

pairs=(
  "0 1"
  "1 2"
  "2 4"
)

for pair in "${pairs[@]}"; do
    read -r F C <<< "$pair"

    for M in 0 1 2 3; do
        echo "Running eval_MLP.py with F=$F, C=$C, M=$M"
        F=$F C=$C M=$M python eval_MLP.py
    done
done
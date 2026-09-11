#!/bin/bash

pairs=(
  "0 1"
  "1 2"
  "2 4"
)

for pair in "${pairs[@]}"; do
    read -r F C <<< "$pair"
    echo "Running eval_RF.py with F=$F, C=$C"
    F=$F C=$C python eval_RF.py
done

#!/bin/bash

models=("BPR" "LightGCN" "DeepFM" "DSSM" "FM" "WideDeep" "NGCF" "SGL")
port=5678

for model in "${models[@]}"; do
    port=$((port + 1))  # 更新 port 值
    python run_recbole.py --model="$model" --port="$port" > "${model}.log" 2>&1 &
done
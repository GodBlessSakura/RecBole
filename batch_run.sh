#!/bin/bash

models=("BPR" "LightGCN" "DeepFM" "DSSM" "FM" "WideDeep" "NGCF" "SGL")
port=5678

for model in "${models[@]}"; do
    port=$((port + 1))  # 更新 port 值
    python run_recbole.py --model="$model" --port="$port" > "${model}.log" 2>&1 &
done

# 在四种backbone上热数据的表现
nohup python run_recbole.py --model=LightGCN --port=6777  --dataset=CiteULike >> CiteULike_LGNwarm.log 2>&1 &
nohup python run_recbole.py --model=BPR --port=6778  --dataset=CiteULike >> CiteULike_BPRwarm.log 2>&1 &
nohup python run_recbole.py --model=SGL --port=6779  --dataset=CiteULike >> CiteULike_SGLwarm.log 2>&1 &
nohup python run_recbole.py --model=NGCF --port=6780  --dataset=CiteULike >> CiteULike_NGCFwarm.log 2>&1 &
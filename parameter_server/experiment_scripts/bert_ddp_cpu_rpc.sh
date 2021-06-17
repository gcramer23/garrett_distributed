#!/bin/sh

cd "$(dirname "$0")"
cd ..

python -u launcher.py \
    --master_addr="localhost" \
    --master_port="29500" \
    --trainer="DdpBertTrainer" \
    --ntrainer=2 \
    --ncudatrainer=0 \
    --filestore="/tmp/tmpn_k_8so02" \
    --server="AverageParameterServer" \
    --nserver=1 \
    --ncudaserver=0 \
    --rpc_timeout=1000 \
    --backend="gloo" \
    --epochs=1 \
    --batch_size=1 \
    --data="BertCommonsenseData" \
    --model="BertCommonsenseModel" \
    --data_config_path="configurations/data_configurations.json" \
    --model_config_path="configurations/model_configurations.json" \
    --ddp_hook="rpc_hook"
#!/bin/sh

cd "$(dirname "$0")"
cd ..

python -u launcher.py \
    --master_addr="localhost" \
    --master_port="29500" \
    --trainer="DdpTrainer" \
    --ntrainer=0 \
    --ncudatrainer=2 \
    --filestore="/tmp/tmpn_k_8so02" \
    --server="AverageParameterServer" \
    --nserver=0 \
    --ncudaserver=1 \
    --rpc_timeout=30 \
    --backend="nccl" \
    --epochs=1 \
    --batch_size=1 \
    --data="DummyData" \
    --model="DummyModelSparse" \
    --data_config_path="configurations/data_configurations.json" \
    --model_config_path="configurations/model_configurations.json" \
    --preprocess_data="preprocess_dummy_data" \
    --create_criterion="cel" \
    --create_ddp_model="basic_ddp_model" \
    --create_optimizer="sgd_optimizer" \
    --iteration_step="basic_iteration_step" \
    --hook_state="BasicHookState" \
    --ddp_hook="sparse_rpc_hook" \
    --lr=1e-4
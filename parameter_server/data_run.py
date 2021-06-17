import os
import argparse

def run(nt, ns, nct, ncs, hook, server, backend, file_prefix):
    os.system(
        f"""python launcher.py --master_addr="localhost" \
        --master_port="29500" \
        --trainer="DdpTrainer" \
        --ntrainer={nt} \
        --ncudatrainer={nct} \
        --filestore="/tmp/tmpn_k_8so02" \
        --server={server} \
        --nserver={ns} \
        --ncudaserver={ncs} \
        --rpc_timeout=120 \
        --backend={backend} \
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
        --ddp_hook={hook} \
        --lr=1e-4 \
        --prefix_metrics_output_name={file_prefix}
        """
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_trainers",
        type=str,
        default=3,
        help="maximum number of trainers that will be used in an experiment run"
    )

    parser.add_argument(
        "--max_servers",
        type=str,
        default=1,
        help="maximum number of servers that will be used in an experiment run"
    )

    args = parser.parse_args()

    hooks = {"sparse_rpc_hook", "rpc_hook"}
    servers = {"AverageParameterServer", "AverageBatchParameterServer"}
    rpcs = {"cpu", "cuda"}

    for rpc in rpcs:
        for hook in hooks:
            for server in servers:
                nt = 1
                while nt <= args.max_trainers:
                    ns = 1
                    while ns <= args.max_servers:
                        file_prefix = f"{hook}_{server}".lower()
                        if "sparse" in hook:
                            backend = "nccl"
                        else:
                            backend = "gloo"
                        run(nt, ns, 0, 0, hook, server, backend, file_prefix)
                        run(0, 0, nt, ns, hook, server, backend, file_prefix)
                        ns *= 2
                    nt *= 2

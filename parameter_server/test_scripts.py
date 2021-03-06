import subprocess
from os.path import join
from pathlib import Path

script_dir = join(
    Path(__file__).parent, "experiment_scripts"
)
encoding = 'utf-8'


def run_script(script_name):
    # runs the script and asserts that there are no errors
    p = subprocess.run(
        ["bash", f"{join(script_dir,script_name)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    error = p.stderr.decode(encoding)
    assert not error


def test_ddp_nccl_allreduce():
    run_script("ddp_nccl_allreduce.sh")


def test_ddp_cpu_sparse_rpc_nccl_allreduce():
    run_script("ddp_cpu_sparse_rpc_nccl_allreduce.sh")


def test_ddp_cuda_sparse_rpc_nccl_allreduce():
    run_script("ddp_cuda_sparse_rpc_nccl_allreduce.sh")


def test_ddp_batch_cpu_sparse_rpc_nccl_allreduce():
    run_script("ddp_batch_cpu_sparse_rpc_nccl_allreduce.sh")


def test_ddp_batch_cuda_sparse_rpc_nccl_allreduce():
    run_script("ddp_batch_cuda_sparse_rpc_nccl_allreduce.sh")


def test_ddp_cpu_rpc():
    run_script("ddp_cpu_rpc.sh")


def test_ddp_cuda_rpc():
    run_script("ddp_cuda_rpc.sh")


def test_ddp_batch_cpu_rpc():
    run_script("ddp_batch_cpu_rpc.sh")


def test_ddp_batch_cuda_rpc():
    run_script("ddp_batch_cuda_rpc.sh")

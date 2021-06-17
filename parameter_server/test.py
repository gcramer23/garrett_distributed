import os

import torch
import torch.distributed as c10d
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP


def sparse_tensor_to_rpc_format(sparse_tensor):
    sparse_tensor = sparse_tensor.coalesce()
    return [sparse_tensor.indices(), sparse_tensor.values(), sparse_tensor.size()]


def sparse_rpc_format_to_tensor(sparse_rpc_format):
    return torch.sparse_coo_tensor(
        sparse_rpc_format[0], sparse_rpc_format[1], sparse_rpc_format[2]
    ).coalesce()


class Server:

    @staticmethod
    @rpc.functions.async_execution
    def identity(tensor):
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=False)

    def forward(self, x):
        return self.embedding(x)


def run_trainer(rank, rref, cuda_rpc):

    def basic_hook(state, bucket):

        def callback(fut):
            tensor = fut.wait()
            if type(tensor) is list:
                tensor = sparse_rpc_format_to_tensor(tensor)
            if not cuda_rpc:
                tensor = tensor.cuda(rank)
            return [tensor]

        tensor = bucket.get_tensor()
        if not cuda_rpc:
            tensor = tensor.cpu()
        if tensor.is_sparse:
            tensor = sparse_tensor_to_rpc_format(tensor)
        fut = rref.rpc_async().identity(tensor).then(callback)
        return fut

    model = Model().cuda(rank)
    store = c10d.FileStore("/tmp/tmpn_k_8so02", 1)
    process_group = c10d.ProcessGroupGloo(store, rank, 1)
    ddp_model = DDP(model, device_ids=[rank], process_group=process_group)
    ddp_model.register_comm_hook(None, basic_hook)
    loss_fn = nn.MSELoss()
    input = torch.randint(5, (10, 10)).cuda(rank)
    loss = loss_fn(ddp_model(input), input.to(torch.float))
    print(loss)


def run_test(rank, cuda_rpc):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    opts = TensorPipeRpcBackendOptions()
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=3,
            rpc_backend_options=opts
        )
        rref = rpc.remote("server", Server)
        rpc.rpc_async(
            "trainer",
            run_trainer,
            args=(0, rref, cuda_rpc,)
        )
    elif rank == 0:
        if cuda_rpc:
            opts.set_device_map("server", {rank: 1})
        rpc.init_rpc(
            "trainer",
            rank=rank,
            world_size=3,
            rpc_backend_options=opts
        )
    else:
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=3,
            rpc_backend_options=opts
        )
    rpc.shutdown()


if __name__ == "__main__":
    cuda_rpc = True
    mp.spawn(
        run_test,
        nprocs=3,
        args=(cuda_rpc,),
        join=True
    )

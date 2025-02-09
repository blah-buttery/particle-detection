import os
import torch
import torch.distributed as dist

def setup_ddp(rank, world_size):
    """
    Sets up the Distributed Data Parallel (DDP) environment.

    This function configures the necessary environment variables for DDP, 
    such as the master address and port, and initializes the process group.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes participating in DDP.

    Raises:
        RuntimeError: If the DDP initialization fails.
    """
    print(f"Setting up DDP for rank {rank} with world size {world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

def cleanup_ddp():
    """
    Cleans up the Distributed Data Parallel (DDP) environment.

    This function destroys the process group if it has been initialized. 
    It is safe to call even if DDP was not initialized.

    Raises:
        AssertionError: If the process group cleanup fails.
    """
    if torch.distributed.is_initialized():
        print("Cleaning up DDP")
        torch.distributed.destroy_process_group()
    else:
        print("DDP process group is not initialized, skipping cleanup")


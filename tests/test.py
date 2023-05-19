import os
import torch
import dist_op
import torch.distributed as dist
import pickle

rank = int(os.getenv('RANK'))
local_rank = int(os.getenv('LOCAL_RANK'))
world_size = int(os.getenv('WORLD_SIZE'))
print(f'Rank/World Size: {rank}/{world_size}')
torch.distributed.init_process_group(
    backend='nccl', init_method='env://', world_size=world_size, rank=rank)
torch.distributed.barrier()
torch.cuda.set_device(local_rank)

assert world_size == 2

def set_fp8_mode(enabled):
    # TODO
    pass

# test case 1
# Check uint8 all reduce
xs = [torch.tensor([100, 200, 300], dtype=torch.uint8, device='cuda'),
      torch.tensor([130, 220, 307], dtype=torch.uint8, device='cuda')]
target = xs[0] + xs[1]

x = xs[rank].clone()

set_fp8_mode(enabled=False)
torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)

if torch.all(x == target, dtype=torch.uint8, device='cuda'):
    print('[passed] uint8 all reduce passed')
else:
    print('[failed] uint8 all reduce failed')

# test case 2
# Check fp8e4m3 all reduce
# E4M3: 4 exponent bits, 3 mantissa bits
xs = [torch.tensor([0b01001000, 0b01011000], dtype=torch.uint8, device='cuda'),
      torch.tensor([0b01010000, 0b01011000], dtype=torch.uint8, device='cuda')]
target = torch.tensor([0b01011100, 0b01101000], dtype=torch.uint8, device='cuda')
x = xs[rank].clone()

set_fp8_mode(enabled=True)
torch.distributed.all_reduce(x, op=dist.ReduceOp.SUM)

if torch.all(x == target, dtype=torch.uint8, device='cuda'):
    print('[passed] fp8e4m3 all reduce passed')
else:
    print('[failed] fp8e4m3 all reduce failed')

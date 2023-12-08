import os
import math
from VPTDataset import VPTDataset

dataset = VPTDataset()

batch_size = 200
num_batches = math.ceil(len(dataset) / batch_size)

for batch_idx in range(num_batches):
    os.sys(f'sbatch python train.py --batch-size {batch_size} --batch-idx {batch_idx}')
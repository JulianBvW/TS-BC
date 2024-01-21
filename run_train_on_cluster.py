import os
import math
from VPTDataset import VPTDataset

dataset = VPTDataset()

batch_size = 40
num_batches = math.ceil(len(dataset) / batch_size)
save_dir = '/ssd005/projects/minecraft_2024/latents/'

print('Total Examples:', num_batches * batch_size)
print('Batches:', num_batches)
print('---')

for batch_idx in range(num_batches):
    launch_cmd = f'sbatch run_latent_computation.sh {batch_size} {batch_idx} {save_dir}'
    print(launch_cmd)
    os.system(launch_cmd)
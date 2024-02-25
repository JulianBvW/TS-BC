#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=30G
#SBATCH --qos=m3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mc_latents
#SBATCH --output=/h/andrei/TS-BC/logs_feb19_2024/mc_latents_%j.out
#SBATCH --error=/h/andrei/TS-BC/logs_feb19_2024/mc_latents_%j.error

batch_size=$1
batch_idx=$2
save_dir=$3

echo "batch_size: $batch_size"
echo "batch_idx: $batch_idx"
echo "save_dir: $save_dir"

source /h/andrei/.bashrc
conda activate mc39
. java8_setup.sh

python train.py --batch-size=$batch_size --batch-idx=$batch_idx --save-dir=$save_dir

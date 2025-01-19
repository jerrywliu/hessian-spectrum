#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=m1266
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15

conda activate /pscratch/sd/j/jwl50/hessian-spectrum/language_models/.pyenv
torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_bfs.py
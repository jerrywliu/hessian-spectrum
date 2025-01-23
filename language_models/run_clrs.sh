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
# Tasks
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_bfs.py
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_dfs.py
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_heapsort.py
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_minimum.py
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_binary_search.py
# torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_bubble_sort.py
torchrun --standalone --nproc_per_node=4 train_clrs.py config/train_gpt2_nano_clrs_insertion_sort.py

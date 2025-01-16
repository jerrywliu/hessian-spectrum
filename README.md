# Why Transformers Need Adam: A Hessian Perspective
This repository contains PyTorch implementations of blockwise and full Hessian spectrum estimation of large-scale neural nets via the Stochastic Lanscoz Quadrature method.  Check out more descriptions in our paper https://arxiv.org/abs/2402.16788.

## How to use 

Here we provide two examples for Hessian spectrum estimation. One for various models on ImageNet and one for GPT2 on Openwebtext.

### For vision models 

Set up the Python environment using Anaconda with the provided `vision_models/environment.yml` file.

```
conda env create -f vision_models/environment.yml
conda activate cvmodels
```

Run the code for hessian spectrum estimation. 

```
bash vision_models/run.sh
```

### For language models 

Set up the Python environment using Anaconda with the provided `language_models/environment.yml` file.

```
conda env create -f vision_models/environment.yml
conda activate gpt2
```

Run the code for hessian spectrum estimation. 

```
cd language_models
conda activate /pscratch/sd/j/jwl50/hessian-spectrum/language_models/.pyenv
bash run_gpt2.sh
```

Run the code on 4 gpus, 1 node:
```
torchrun --standalone --nproc_per_node=4 train_gpt2.py config/train_gpt2_nano_shakespeare.py
torchrun --standalone --nproc_per_node=4 train_gpt2.py config/train_gpt2_small_openwebtext.py
```
CLRS:
```
torchrun --standalone --nproc_per_node=4 train_gpt2.py config/train_gpt2_nano_clrs_bfs.py
torchrun --standalone --nproc_per_node=4 train_gpt2.py config/train_gpt2_nano_clrs_dfs.py
```

TODO run the code on 4 GPUs and 4 nodes:
```
torchrun --nnodes=4 --nproc_per_node=4 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_gpt2.py
```
Make sure to replace `$NODE_RANK`, `$MASTER_ADDR`, and `$MASTER_PORT` with appropriate values for your setup.

## Remark

Our code for spectrum estimation could be easily plugged into most public implementations for neural-net training. To assure the accuracy of estimation, please remember to remove all the randomness in the forward and backward passes (e.g., data shuffling and dropout).  Since Flash Attention does not support the calculation of Hessian-vector product, please make sure  all attention blocks are implemented in a naive way (as in the GPT2 example above). See more tips in Appendix C.1 in the paper. 


## Acknowledgements

The above code is heavily based on the code base of [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) and [NanoGPT](https://github.com/karpathy/nanoGPT/) . 

## Citation

If you find this code helpful, please cite our paper in the following format.

```
@article{zhang2024why,
  title     = {Why Transformers Need Adam: A Hessian Perspective},
  author    = {Zhang, Yushun and Congliang, Chen and Tian, Ding and Ziniu, Li and Sun, Ruoyu and Luo, Zhi-Quan},
  booktitle = {arXiv preprint arXiv:2402.16788},
  year      = {2024},
}
```

# for a given model checkpoint, compute the spectrum of the hessian

import torch
import numpy as np
import os
import pickle
from contextlib import nullcontext

from model import GPT, GPTConfig
from hessian_spectrum import Hessian


# checkpoint_path = "/pscratch/sd/j/jwl50/hessian-spectrum/language_models/out-gpt2/gptnano_clrs_binary_search/ckpt_epoch_0.00_iter_5.pt"
checkpoint_path = "/pscratch/sd/j/jwl50/hessian-spectrum/language_models/out-gpt2/gptnano_clrs_binary_search/ckpt_epoch_0.02_iter_4000.pt"

# default config values (overridden by command line arguments)

# data
dataset = "clrs"
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
bias = False
dropout = 0.1
flash_attn = False
device = "cuda"
dtype = "float32"

# hessian
sample_layer = []
comment = ""

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

# make a dict of model args
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50257,
    dropout=dropout,
    flash_attn=flash_attn,
    device=device,
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# load the checkpoint
checkpoint = torch.load(checkpoint_path)
checkpoint_model_args = checkpoint["model_args"]
for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
    model_args[k] = checkpoint_model_args[k]
model.load_state_dict(checkpoint["model"])
model.to(device)
# load the data
data_dir = os.path.join("data", dataset)
train_prompts = np.memmap(
    os.path.join(data_dir, "train_prompts.bin"), dtype=np.uint16, mode="r"
)
train_targets = np.memmap(
    os.path.join(data_dir, "train_targets.bin"), dtype=np.uint16, mode="r"
)
val_prompts = np.memmap(
    os.path.join(data_dir, "val_prompts.bin"), dtype=np.uint16, mode="r"
)
val_targets = np.memmap(
    os.path.join(data_dir, "val_targets.bin"), dtype=np.uint16, mode="r"
)

with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
prompt_lengths = meta["prompt_lengths"]
target_lengths = meta["target_lengths"]

# compute the spectrum
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
use_minibatch = True
gradient_accumulation_steps = 60
hessian = Hessian(
    model,
    m=100,
    num_v=1,
    ckpt_iteration=0,
    train_data=train_prompts,
    batch_size=batch_size,
    block_size=block_size,
    ctx=ctx,
    use_minibatch=use_minibatch,
    gradient_accumulation_steps=gradient_accumulation_steps,
    device=device,
    sample_layer=sample_layer,
    comment=comment,
)
hessian.get_spectrum(layer_by_layer=True)
hessian.load_curve(layer_by_layer=True)

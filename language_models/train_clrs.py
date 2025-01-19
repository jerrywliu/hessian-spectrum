"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_gpt2.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import json
import hessian_spectrum
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb
import logging
import sys

from model import GPTConfig, GPT


# ipdb.set_trace()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
resume_dir = None
eval_interval = 20
ckpt_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval

init_from = "scratch"  #'resume'# 'scratch' or 'resume' or 'gpt2*'

load_iter = 0

# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# epochs
epochs = 10
resume_epoch = 0
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
seed = 1337
comment = ""
use_minibatch = True
layer_by_layer = False
flash_attn = False
sample_layer = []
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = "float32"  # 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

save_dir = "log_gpt2/"


# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


os.makedirs(save_dir, exist_ok=True)


# io_utils.save_code(save_dir)

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(
        project="hessian-spectrum",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": dropout,
            "bias": bias,
            "flash_attn": flash_attn,
        },
    )

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}[dtype]
print(f"ptdtype = {ptdtype}")
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


# poor man's data loader
data_dir = os.path.join("data", dataset)

# Load data
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

# Load metadata to get lengths
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
prompt_lengths = meta["prompt_lengths"]
target_lengths = meta["target_lengths"]


def get_batch(split):
    # Get the appropriate data sources
    prompts = train_prompts if split == "train" else val_prompts
    targets = train_targets if split == "train" else val_targets

    # Sample batch_size examples
    batch_sequences = []
    batch_masks = []

    while len(batch_sequences) < batch_size:
        # Randomly select an example index
        ex_idx = torch.randint(len(prompt_lengths), (1,)).item()
        p_len = prompt_lengths[ex_idx]
        t_len = target_lengths[ex_idx]

        # Skip if total length exceeds block size
        if p_len + t_len > block_size:
            continue

        # Get sequences
        prompt_start = sum(prompt_lengths[:ex_idx])
        target_start = sum(target_lengths[:ex_idx])
        prompt = prompts[prompt_start : prompt_start + p_len]
        target = targets[target_start : target_start + t_len]

        # Convert to tensors
        prompt = torch.from_numpy(prompt.astype(np.int64))
        target = torch.from_numpy(target.astype(np.int64))

        # Concatenate prompt and target
        sequence = torch.cat([prompt, target])
        # Create mask (0 for prompt, 1 for target tokens)
        mask = torch.cat([torch.zeros(p_len), torch.ones(t_len)])

        # Pad to block_size
        if sequence.size(0) < block_size:
            sequence = F.pad(sequence, (0, block_size - sequence.size(0)), value=0)
            mask = F.pad(mask, (0, block_size - mask.size(0)), value=0)

        batch_sequences.append(sequence)
        batch_masks.append(mask)

    # Stack batches
    batch_sequences = torch.stack(batch_sequences)
    batch_masks = torch.stack(batch_masks)

    if device_type == "cuda":
        batch_sequences = batch_sequences.pin_memory().to(device, non_blocking=True)
        batch_masks = batch_masks.pin_memory().to(device, non_blocking=True)
    else:
        batch_sequences = batch_sequences.to(device)
        batch_masks = batch_masks.to(device)

    return batch_sequences, batch_masks


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
current_epoch = 0


# load_iter = int(os.environ.get("LOAD_ITER"))
print("load_iter = ", load_iter, "loading ..", load_iter)

if load_iter == 0:
    init_from = "scratch"
else:
    init_from = "resume"


# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    flash_attn=flash_attn,
    device=device,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


elif init_from == "resume":

    if resume_dir == None:
        resume_dir = out_dir
    print(f"Resuming training from {resume_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(resume_dir, "ckpt" + str(load_iter) + ".pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]

elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)


if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None  # free up memory


# wrap model into DDP container
if ddp:
    # model = DDP(model, device_ids=[ddp_local_rank])
    model = torch.nn.DataParallel(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    t_eval = time.time()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        sequence_correct = 0
        total_sequences = 0
        token_correct = 0
        total_tokens = 0

        for k in range(eval_iters):
            sequences, masks = get_batch(split)
            targets = torch.cat(
                [sequences[:, 1:], torch.zeros_like(sequences[:, :1])], dim=1
            )

            with ctx:
                logits, loss = model(sequences, targets=targets, attention_mask=masks)

            # Calculate loss
            losses[k] = loss.item()

            # Get predictions
            pred = logits.argmax(dim=-1)  # [batch, sequence_length]

            # Calculate accuracy only on target tokens
            valid_tokens = masks == 1

            # Shift targets and masks for accuracy calculation (we predict next token)
            shifted_sequences = sequences[:, 1:]  # Remove first token
            shifted_masks = masks[:, :-1].to(torch.bool)  # Remove last mask position
            pred = pred[:, :-1]  # Remove last prediction

            # Sequence-level accuracy (only count sequences where all tokens are correct)
            seq_correct = ((pred == shifted_sequences) | ~shifted_masks).all(dim=1)
            sequence_correct += seq_correct.sum().item()
            total_sequences += sequences.size(0)

            # Token-level accuracy (only on target tokens)
            token_matches = (pred == shifted_sequences) & shifted_masks
            token_correct += token_matches.sum().item()
            total_tokens += shifted_masks.sum().item()

        out[f"{split}/loss"] = losses.mean()
        out[f"{split}/sequence_accuracy"] = sequence_correct / total_sequences
        out[f"{split}/token_accuracy"] = token_correct / total_tokens

    print(f"validation done. time used = {time.time() - t_eval}")
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


best_val_loss = float("inf")  # Initialize with infinity so first loss will be better


def save_checkpoint(raw_model, optimizer, current_epoch, iter_num, val_loss):
    global best_val_loss
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "current_epoch": current_epoch,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "config": config,
    }

    # Save the checkpoint
    checkpoint_path = os.path.join(
        out_dir, f"ckpt_epoch_{current_epoch:.2f}_iter_{iter_num}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint_path = os.path.join(out_dir, f"best_ckpt.pt")
        torch.save(checkpoint, best_checkpoint_path)
        print(
            f"New best validation loss: {best_val_loss}. Saving checkpoint to {best_checkpoint_path}"
        )


# logging

sequences, masks = get_batch("train")  # fetch the very first batch


def plot_hessian():

    # batch_size = 8
    # gradient_accumulation_steps = 60
    use_minibatch = True

    hessian = hessian_spectrum.Hessian(
        model,
        ckpt_iteration=load_iter,
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
    # Wait for the hessian to finish
    time.sleep(10)
    hessian.load_curve(layer_by_layer=True)

    # hessian.get_spectrum(layer_by_layer=False)
    # hessian.load_curve(layer_by_layer=False)


def train():
    global iter_num, X, Y, current_epoch, best_val_loss
    "the following is for training"
    # training loop
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0

    while iter_num < max_iters:
        # Calculate equivalent epochs
        samples_processed = iter_num * batch_size * gradient_accumulation_steps
        current_epoch = samples_processed / (len(train_prompts) - block_size)

        if iter_num % log_interval == 0 and master_process:
            logging.info(f"iter {iter_num}: epoch {current_epoch:.2f}")

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:

            metrics = estimate_loss()
            logging.info(
                f"step {iter_num}: train loss {metrics['train/loss']:.4f}, "
                f"val loss {metrics['val/loss']:.4f}, "
                f"train seq acc {metrics['train/sequence_accuracy']:.2%}, "
                f"val seq acc {metrics['val/sequence_accuracy']:.2%}, "
                f"train token acc {metrics['train/token_accuracy']:.2%}, "
                f"val token acc {metrics['val/token_accuracy']:.2%}"
            )

            # Log to wandb
            wandb.log(
                {
                    **metrics,  # This will log all metrics with their original names
                    "lr": lr,
                    "epoch": current_epoch,
                    "iter": iter_num,
                }
            )

        if (
            master_process
            and iter_num > 0
            and iter_num % ckpt_interval == 0
            or iter_num
            in [
                round(max_iters * 0.001),
                round(max_iters * 0.025),
                round(max_iters * 0.05),
                round(max_iters * 0.075),
                round(max_iters * 0.1),
            ]
        ):

            losses = estimate_loss()
            # save ckpt
            save_checkpoint(
                raw_model, optimizer, current_epoch, iter_num, losses["val/loss"]
            )

        if iter_num == 0 and eval_only:
            break

        # forward backward update
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                sequences, masks = get_batch("train")
                logits, loss = model(
                    sequences,
                    targets=torch.cat(
                        [sequences[:, 1:], torch.zeros_like(sequences[:, :1])], dim=1
                    ),
                    attention_mask=masks,
                )
                loss = loss / gradient_accumulation_steps

            # immediately async prefetch next batch
            X, Y = get_batch("train")

            # backward pass
            scaler.scale(loss).backward()

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            logging.info(
                f"epoch {current_epoch:.2f}, iter {iter_num}: train loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

            # Log training metrics
            if master_process:
                wandb.log(
                    {
                        "train/iter_loss": lossf,
                        "perf/mfu": running_mfu * 100,
                        "perf/iter_time_ms": dt * 1000,
                    }
                )

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()

    # Close wandb run
    if master_process:
        wandb.finish()


plot_hessian()

train()

plot_hessian()

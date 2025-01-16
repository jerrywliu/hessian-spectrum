# data
dataset = "clrs/bfs"
batch_size = 16
block_size = 256
gradient_accumulation_steps = 4

# Keep these the same
learning_rate = 6e-4
min_lr = 6e-5
weight_decay = 1e-1
dropout = 0.0

# GPT-Nano
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# this makes total number of tokens be 300B
max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 20
eval_iters = 200  # how many samples used for calulating validation loss
log_interval = 10
ckpt_interval = 1000

# optimizer
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95  # 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 100
min_lr = 3e-5
dtype = "float32"
# use_minibatch = True
# plot_histogram = False

init_from = "scratch"
load_iter = 0

sample_layer = [
    "module.transformer.h.5.attn.wq.weight",
    "module.transformer.h.5.attn.wk.weight",
    "module.transformer.h.5.attn.wv.weight",
    "module.transformer.h.5.attn.wo.weight",
    "module.transformer.h.5.mlp.c_fc.weight",
    "module.transformer.h.5.mlp.c_proj.weight",
    "module.transformer.wte.weight",
]


comment = "gptnano_clrs_bfs"  # only used in tensorboard and hessian class
save_dir = "log_gpt2/" + comment
out_dir = "out-gpt2/" + comment  # save ckpt

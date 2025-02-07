import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, ReasoningState

# Configuration (can be overridden by config files or command line)
out_dir = 'out-symbolic'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'symbolic-reasoning'
wandb_run_name = 'lcm-gpt'

# data
dataset = 'symbolic_shakespeare'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 64  # Max concept sequence length

# model
n_layer = 6
n_head = 4
n_embd = 128
dropout = 0.1
bias = False
# concept_vocab_size and relation_vocab_size will be set from meta.pkl

# adamw optimizer
learning_rate = 1e-3
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 200
lr_decay_iters = max_iters
min_lr = 1e-4

# DDP settings
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16'
compile = True

# LCM Specific
consistency_steps = 3  # Now used for generating multi-step targets
lcm_loss_weight = 0.5  # Weight of the consistency loss
concept_threshold = 0.5  # Threshold for adding concepts
relation_threshold = 0.5  # Threshold for adding relations


# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
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
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading, now creating concept and relation targets
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size and relation_vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    relation_vocab_size = len(meta['relation_stoi']) # Get size from stoi
    stoi = meta['stoi']
    itos = meta['itos']
    relation_stoi = meta['relation_stoi']
    relation_itos = meta['relation_itos']

    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    print(f"found relation_vocab_size = {relation_vocab_size}")

# model init.  Note that concept_vocab_size and relation_vocab_size are now REQUIRED
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout, relation_vocab_size = relation_vocab_size)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # resume training from a checkpoint.
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
     # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'relation_vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, concept_targets, relation_targets = get_batch(split) # Call get_batch, unpack
            with ctx:
                concept_probs, relation_probs, loss = model(initial_state, concept_targets=concept_targets, relation_targets=relation_targets) # Pass targets
            losses[k] = loss.item()
        out[split] = losses.mean()
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
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)



def get_batch(split):
    data = train_data if split == 'train' else val_data
    max_seq_len = block_size
    ix = torch.randint(len(data) - max_seq_len - consistency_steps, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])

    # Create target concept and relation probabilities
    concept_targets = []
    relation_targets = []
    for i in ix:
        state = ReasoningState(knowledge_graph=knowledge_graph)
        temp_concept_targets = []
        temp_relation_targets = []

        for j in range(consistency_steps):
            current_concept_id = data[i + j].item()  # Get current concept

            # Update the state using the *true* next concept (supervised learning)
            state.transition([current_concept_id], concept_probs=None, relation_probs=None, concept_threshold=0, relation_threshold=0)

            # Create concept target: 1 for concepts in the state, 0 otherwise
            concept_target = torch.zeros(meta_vocab_size, dtype=torch.float) #float for bce loss
            for concept_id in state.concepts:
                if concept_id < meta_vocab_size:  # Safety check
                    concept_target[concept_id] = 1.0
            temp_concept_targets.append(concept_target)

            # Create relation target:
            relation_target = torch.zeros((block_size, config.relation_vocab_size), dtype=torch.float)
            # Iterate over possible pairs within the *current* working memory
            for idx1, concept_id1 in enumerate(state.working_memory):
                    for idx2, concept_id2 in enumerate(state.working_memory):
                        if idx1 != idx2:
                            #The concept_ids needs to be strings:
                            subj = knowledge_graph.itos_concepts[concept_id1]
                            obj = knowledge_graph.itos_concepts[concept_id2]
                            for rel_idx in range(config.relation_vocab_size):
                                rel_id = knowledge_graph.itos_relations[rel_idx] #get the relation id from index
                                if (subj, rel_id, obj) in state.relations:
                                    relation_target[idx1,rel_idx] = 1.0 #Fill with ones
            temp_relation_targets.append(relation_target)


        concept_targets.append(torch.stack(temp_concept_targets))
        relation_targets.append(torch.stack(temp_relation_targets))


    concept_targets = torch.stack(concept_targets)
    relation_targets = torch.stack(relation_targets)



    if device_type == 'cuda':
        x, concept_targets, relation_targets = x.pin_memory().to(device, non_blocking=True), concept_targets.pin_memory().to(device, non_blocking=True), relation_targets.pin_memory().to(device, non_blocking=True)
    else:
        x, concept_targets, relation_targets = x.to(device), concept_targets.to(device), relation_targets.to(device)
    return x, concept_targets, relation_targets


# --- LCM Loss Function ---
def lcm_loss(concept_probs, relation_probs, concept_targets, relation_targets, consistency_steps, lcm_loss_weight):

    batch_size = concept_probs.size(0)
    total_loss = 0.0

    # 1. Direct Prediction Loss (Binary Cross-Entropy)
    direct_concept_loss = F.binary_cross_entropy(concept_probs, concept_targets[:, 0]) #First element
    direct_relation_loss = F.binary_cross_entropy(relation_probs, relation_targets[:, 0])
    total_loss += direct_concept_loss + direct_relation_loss

    # 2. Consistency Loss (KL Divergence or Binary Cross-Entropy - we'll stick with BCE)
    consistency_loss = 0.0
    for step in range(1, consistency_steps):

        consistency_loss += F.binary_cross_entropy(concept_probs, concept_targets[:, step])
        consistency_loss += F.binary_cross_entropy(relation_probs, relation_targets[:, step])


    total_loss += lcm_loss_weight * (consistency_loss / (consistency_steps - 1))
    return total_loss

#Define Knowledge Graph
from data.symbolic_prepare_shakespeare import OfflineWikidataKnowledgeGraph
knowledge_graph = OfflineWikidataKnowledgeGraph('data/symbolic_shakespeare/wikidata_subset.tsv')


# training loop
X, concept_targets, relation_targets  = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            initial_state = ReasoningState(knowledge_graph=knowledge_graph)  # Start with a fresh state each time
            concept_probs, relation_probs, loss = model(initial_state, concept_targets=concept_targets[:,0], relation_targets=relation_targets[:,0])  # Get initial predictions
            # --- Multi-Step Forward Passes for Consistency Loss ---
            for step in range(1, consistency_steps):
                # 1. Simulate state transition (using current probabilities, *not* rules)
                with torch.no_grad():  # Don't track gradients during state transitions
                    initial_state.transition(concept_probs[0], relation_probs[-1], concept_threshold, relation_threshold) #Use last timestep output

                # 2. Forward pass with updated state
                next_concept_probs, next_relation_probs, _ = model(initial_state) #No targets here

                # 3. Accumulate logits (we'll compute the loss later)
                # Pad with zeros if working memory is empty:
                if next_concept_probs.numel() == 0:
                    next_concept_probs = torch.zeros_like(concept_probs)

                concept_probs = torch.cat([concept_probs, next_concept_probs], dim=0)
                relation_probs = torch.cat([relation_probs, next_relation_probs[-1:, :]], dim=0)  # Only last timestep

            loss = lcm_loss(concept_probs, relation_probs, concept_targets, relation_targets, consistency_steps, lcm_loss_weight) # Use new loss function
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, concept_targets, relation_targets  = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
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
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)  # Needs to be adapted
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

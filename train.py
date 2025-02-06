import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, ReasoningState  # Import ReasoningState

# -----------------------------------------------------------------------------
# Configuration (can be overridden by config files or command line)
out_dir = 'out-symbolic'
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume', or 'gpt2*'

# wandb logging
wandb_log = False
wandb_project = 'symbolic-reasoning'
wandb_run_name = 'lcm-gpt'

# data
dataset = 'symbolic_shakespeare'  # Or your custom symbolic dataset
gradient_accumulation_steps = 4
batch_size = 16
block_size = 64  # Max concept sequence length

# model
n_layer = 6
n_head = 4
n_embd = 128
dropout = 0.1
bias = False
concept_vocab_size = 1000  # Replace with actual size from meta.pkl

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
consistency_steps = 3
lcm_loss_weight = 0.5
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
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

# poor man's data loader
data_dir = os.path.join('data', dataset)

# --- Data Loading (Modified for Symbolic Data and Action Targets) ---
def get_batch(split):
     # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Instead of block_size, we now use a variable-length sequence of concept IDs.
    # We'll still use a maximum sequence length for batching.
    max_seq_len = block_size  # Or a different value, if needed
    ix = torch.randint(len(data) - max_seq_len - consistency_steps, (batch_size,)) # Leave space for future states
    x = torch.stack([torch.from_numpy((data[i:i+max_seq_len]).astype(np.int64)) for i in ix])

     #Create target as action_ids, targets are now action_ids from the rule-based system
    targets = []
    for i in ix:
        state = ReasoningState(knowledge_graph=knowledge_graph) #Pass the KG
        temp_targets = [] # collect action ids for consistency_steps
        for j in range(consistency_steps):
            current_concept_ids = [data[i + j].item()] #Get current concept
            # Use your rule-based system here to determine the action (Simplified)
            # In reality, this would involve calling state.transition() and getting the action.
            action_id = rule_based_transition(state, current_concept_ids) # You'll need to implement this
            temp_targets.append(action_id)
            state.transition(current_concept_ids, action_id=action_id, knowledge_graph=knowledge_graph) # Advance the state. Pass the action_id
        targets.append(torch.tensor(temp_targets, dtype=torch.long)) #Append only the action_ids
    targets = torch.stack(targets)

    if device_type == 'cuda':
        x, targets = x.pin_memory().to(device, non_blocking=True), targets.pin_memory().to(device, non_blocking=True)
    else:
        x, targets = x.to(device), targets.to(device)
    return x, targets

# --- Rule-Based Transition Function (with Knowledge Graph Access) ---
def rule_based_transition(state, concept_ids):
    """
    Implements the rule-based state transition logic.  This is a simplified
    example and should be customized based on your specific reasoning rules
    and knowledge graph.
    """
    #For simplicity: 0 = add concept, 1 = add relation (if possible), 2 = no-op
    if not state.concepts:
        return 0 #Add the very first concept
    for concept_id in concept_ids:
        if state.knowledge_graph.get_related_concepts(itos[concept_id]):
            return 1 #Add relation, if possible
    return 2 #No-op

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    stoi = meta['stoi']
    itos = meta['itos']
    relation_stoi = meta['relation_stoi']
    relation_itos = meta['relation_itos']
    num_actions = len(relation_stoi) + 1 #Number of relations + 1 (for no-op or adding concept)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    print(f"Number of actions: {num_actions}")

#Define Knowledge Graph (for use in rule_based_transition and ReasoningState)
from data.symbolic_prepare import WikidataKnowledgeGraph
knowledge_graph = WikidataKnowledgeGraph()


# --- Model Initialization (Modified) ---
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=meta_vocab_size, dropout=dropout, num_actions=num_actions)  # Use concept_vocab_size and num_actions

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'num_actions']: #Added num_actions
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    #Removed gpt-2 initialization.

    print(f"Initializing from OpenAI GPT-2 weights: {init_from} is not supported")
    exit()
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
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

# --- Loss Function (LCM-Inspired) ---

def lcm_loss(logits, targets, consistency_steps, lcm_loss_weight):
    """
    Calculates the LCM loss, including both direct prediction and consistency.

    Args:
        logits: Model output logits (batch_size, sequence_length, num_actions).
        targets: Target action IDs (batch_size, consistency_steps).
        consistency_steps: Number of future steps for consistency.
        lcm_loss_weight: Weight of the consistency loss.

    Returns:
        Total loss (scalar).
    """

    batch_size, seq_len, num_actions = logits.size()
    total_loss = 0.0

    # 1. Direct Prediction Loss (Cross-Entropy)
    direct_loss = F.cross_entropy(logits[:, 0, :], targets[:, 0], ignore_index=-1)  # Compare first prediction to first target
    total_loss += direct_loss

    # 2. Consistency Loss (KL Divergence or Cross-Entropy)
    consistency_loss = 0.0
    for step in range(1, consistency_steps):
      #  consistency_loss += F.cross_entropy(logits[:, step, :], targets[:,step], ignore_index=-1) #Option 1: Cross Entropy
        # Option 2: KL Divergence (requires converting logits to probabilities)
        prev_probs = F.softmax(logits[:, step - 1, :], dim=-1)
        curr_probs = F.softmax(logits[:, step, :], dim=-1)
        consistency_loss += F.kl_div(curr_probs.log(), prev_probs, reduction='batchmean') #Compare probability distributions

    total_loss += lcm_loss_weight * (consistency_loss / (consistency_steps -1))

    return total_loss



# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """
    Estimates the loss over the entire train/val sets.  Adapted for the
    symbolic reasoning setup.
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Forward pass (similar to training loop, but without backprop)
            with ctx:
                initial_state = ReasoningState(knowledge_graph=knowledge_graph)  # Start with an empty state
                action_ids, _ = model(initial_state, targets=Y[:,0]) # first action
                logits = torch.zeros((Y.size(0), consistency_steps, num_actions), device = device) #Dummy logit tensor with correct shape.
                logits[:,0,:] = action_ids
                # Get multiple steps
                for step in range(1, consistency_steps):
                     #Simulate state transition
                    for i in range(action_ids.size(0)):
                        rule_based_transition(initial_state, [X[i,step-1].item()]) #Transition
                    next_action_ids, _ = model(initial_state)
                    logits[:, step, :] = next_action_ids
                loss = lcm_loss(logits, Y, consistency_steps, lcm_loss_weight)  # Use the LCM loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
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

# training loop
X, Y = get_batch('train') # fetch the very first batch
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
        losses = estimate_loss()  # You'll need to adapt estimate_loss() too
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
            # Forward pass now takes the ReasoningState
            initial_state = ReasoningState(knowledge_graph=knowledge_graph) # Start with a fresh state each time. Pass the knowledge graph
            action_ids, _ = model(initial_state, targets=Y[:,0]) #Predict the first action
            logits = torch.zeros((Y.size(0), consistency_steps, num_actions), device = device) #Dummy logit tensor with correct shape.
            logits[:,0,:] = action_ids #Store the first action
            # Get multiple steps for consistency loss
            for step in range(1, consistency_steps):
                 #Simulate state transition using the RULE-BASED system
                for i in range(action_ids.size(0)): #Iterate over batch
                    rule_based_transition(initial_state, [X[i,step-1].item()]) #Transition to get to next state, using concept from original input X.
                next_action_ids, _ = model(initial_state) #Predict next action
                logits[:, step, :] = next_action_ids #Store the predicted action

            loss = lcm_loss(logits, Y, consistency_steps, lcm_loss_weight)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train') #Load next batch
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

import os
import time
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-symbolic'
eval_interval = 200
eval_iters = 50
max_iters = 2000
log_interval = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 16
block_size = 64
learning_rate = 1e-3
weight_decay = 1e-2
beta1, beta2 = 0.9, 0.99

# AMP
use_amp = True  # if device == 'cuda'

# If you want wandb:
wandb_log = False
wandb_project = 'symbolic-reasoning'
wandb_run_name = 'lcm-gpt'

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
data_dir = 'data/symbolic_shakespeare'
train_data_path = os.path.join(data_dir, 'train.bin')
val_data_path   = os.path.join(data_dir, 'val.bin')
meta_path       = os.path.join(data_dir, 'meta.pkl')

train_data = np.memmap(train_data_path, dtype=np.uint32, mode='r')
val_data   = np.memmap(val_data_path,   dtype=np.uint32, mode='r')

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

vocab_size = meta['vocab_size']
relation_vocab_size = meta['relation_vocab_size']
stoi = meta['stoi']
itos = meta['itos']
relation_stoi = meta['relation_stoi']
relation_itos = meta['relation_itos']

print(f"vocab_size = {vocab_size}, relation_vocab_size = {relation_vocab_size}")

# -----------------------------------------------------------------------------
# Dataset / Dataloader
# -----------------------------------------------------------------------------
class ConceptDataset(Dataset):
    """
    For each index, we return:
      x: int64 array of shape (block_size,) -> concept IDs
      y: float array of shape (vocab_size,) -> one-hot next concept
    """
    def __init__(self, data, block_size):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        next_id = self.data[idx + self.block_size]
        y = np.zeros((vocab_size,), dtype=np.float32)
        if next_id < vocab_size:
            y[next_id] = 1.0
        return x.astype(np.int64), y

def create_dataloader(split_data):
    ds = ConceptDataset(split_data, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

train_loader = create_dataloader(train_data)
val_loader   = create_dataloader(val_data)

# -----------------------------------------------------------------------------
# Initialize Model
# -----------------------------------------------------------------------------
config = GPTConfig(
    block_size = block_size,
    vocab_size = vocab_size,
    n_layer    = 6,
    n_head     = 4,
    n_embd     = 128,
    dropout    = 0.1,
    bias       = False,
    relation_embd_dim   = 64,
    relation_vocab_size = relation_vocab_size
)
model = GPT(config)
model.to(device)

print(f"Model with {model.get_num_params()/1e6:.2f}M parameters")

optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type='cuda' if device=='cuda' else 'cpu'
)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        data_iter = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type='cuda', enabled=use_amp):
                concept_probs, relation_embed, loss = model(x, concept_targets=y)
            losses.append(loss.item())
        out[split_name] = float(np.mean(losses))
    model.train()
    return out

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
train_iter = iter(train_loader)
t0 = time.time()

for it in range(max_iters):
    # fetch batch
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)

    x = x.to(device)
    y = y.to(device)

    # forward with mixed precision
    with torch.autocast(device_type='cuda', enabled=use_amp):
        concept_probs, relation_embed, loss = model(x, concept_targets=y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    if it % log_interval == 0:
        dt = time.time() - t0
        print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
        t0 = time.time()

    # evaluate
    if it > 0 and it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

os.makedirs(out_dir, exist_ok=True)
ckpt_path = os.path.join(out_dir, 'model_final.pt')
torch.save(model.state_dict(), ckpt_path)
print(f"Saved final checkpoint to {ckpt_path}")

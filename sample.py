"""
Sample from a trained concept-based model that saved only 'model_final.pt' (weights only).
"""

import os
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
init_from = 'resume'  # we are "resuming" from out_dir
out_dir = 'out-symbolic'       # directory with 'model_final.pt'
data_dir = 'data/symbolic_shakespeare'  # directory with meta.pkl, etc.

# Prompt settings
start_prompt = ""   # can be empty or a string
num_samples = 3
max_new_concepts = 20
temperature = 0.8
top_k = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
compile_model = False  # set True if you want torch.compile (PyTorch 2.0+)
# -----------------------------------------------------------------------------

# Force a certain dtype context if GPU
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# 1) Load checkpoint. For 'model_final.pt' we only have a state_dict.
# -----------------------------------------------------------------------------
ckpt_path = os.path.join(out_dir, 'model_final.pt')
checkpoint = torch.load(ckpt_path, map_location=device)  # weights only

# -----------------------------------------------------------------------------
# 2) Attempt to load meta.pkl for model args (vocab_size, block_size, etc.)
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    relation_vocab_size = meta.get('relation_vocab_size', 0)
    # We can pick any defaults for the other config params:
    model_args = dict(
        block_size=64,       # or load from meta if stored
        vocab_size=vocab_size,
        n_layer=6,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
        relation_embd_dim=64,
        relation_vocab_size=relation_vocab_size
    )
    stoi, itos = meta['stoi'], meta['itos']
else:
    print("No meta.pkl found. We'll assume some defaults and skip stoi/itos.")
    model_args = dict(
        block_size=64,
        vocab_size=50000,
        n_layer=6,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
        relation_embd_dim=64,
        relation_vocab_size=0
    )
    stoi, itos = None, None

# Create GPT config from these arguments
gptconf = GPTConfig(**model_args)

# -----------------------------------------------------------------------------
# 3) Build the model and load the weights
# -----------------------------------------------------------------------------
model = GPT(gptconf)
model.load_state_dict(checkpoint)  # 'checkpoint' is a pure state_dict
model.eval()
model.to(device)

if compile_model:
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Helper encode/decode for concept IDs
# -----------------------------------------------------------------------------
def encode_str_to_ids(s: str):
    # minimal fallback: if stoi is None, everything -> 0
    if stoi is None:
        return [0]*len(s) if len(s) > 0 else [0]
    out = []
    for ch in s:
        if ch in stoi:
            out.append(stoi[ch])
        else:
            out.append(0)
    return out

def decode_ids_to_str(ids):
    if itos is None:
        return " ".join(str(i) for i in ids)
    return " ".join(itos[i] for i in ids)

# -----------------------------------------------------------------------------
# 4) Encode the start_prompt into concept IDs
# -----------------------------------------------------------------------------
if len(start_prompt) == 0:
    print("Empty start prompt; sampling from empty context.")
encoded_prompt = encode_str_to_ids(start_prompt)
if len(encoded_prompt) == 0:
    encoded_prompt = [0]  # fallback if empty
prompt_tensor = torch.tensor([encoded_prompt], dtype=torch.long, device=device)  # (1, T)

# -----------------------------------------------------------------------------
# 5) Generate using the model's built-in generate method
# -----------------------------------------------------------------------------
with torch.no_grad():
    with ctx:
        for _ in range(num_samples):
            out = model.generate(
                prompt_tensor.clone(),
                max_new_tokens=max_new_concepts,
                temperature=temperature,
                top_k=top_k
            )
            out_ids = out[0].tolist()  # shape (T + new)
            decoded = decode_ids_to_str(out_ids)
            print("SAMPLE:")
            print(decoded)
            print("---------------")

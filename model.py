"""
Full definition of a GPT Language Model, modified for symbolic reasoning
with an internal state machine (LCM-inspired).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024  # Now refers to the maximum *concept* sequence length
    vocab_size: int = 50304 # Modified: Now refers to concept vocabulary size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    num_actions: int = 3  # Example: Number of discrete actions for state transitions


class ReasoningState:
    """
    Represents the internal state of the reasoning process.
    """
    def __init__(self, concepts=None, relations=None, working_memory=None, knowledge_graph = None):
        self.concepts = concepts if concepts is not None else set()
        self.relations = relations if relations is not None else []
        self.working_memory = working_memory if working_memory is not None else []
        self.max_working_memory_size = 10  # Example parameter.  Adjust as needed.
        self.knowledge_graph = knowledge_graph #Pass your KG to the state.

    def transition(self, new_concept_ids, action_id, knowledge_graph=None):
        """
        Updates the state based on the provided action and new concept IDs.

        Args:
            new_concept_ids: A list of concept IDs from the current input.
            action_id: The ID of the action to perform (from the Transformer's output).
            knowledge_graph: (Optional) Access to a knowledge graph for relation inference.
        """
        if knowledge_graph is not None:
            self.knowledge_graph = knowledge_graph
        # Action 0: Add concept
        if action_id == 0:
            for concept_id in new_concept_ids:
                self.concepts.add(concept_id)
                self.working_memory.append(concept_id)

        # Action 1: Add relation (if possible, based on knowledge graph)
        elif action_id == 1:
             for concept_id in new_concept_ids:
                for existing_concept_id in self.concepts:
                    if self.knowledge_graph.has_relation(existing_concept_id, concept_id):
                        self.relations.append((existing_concept_id, "RELATION_ID", concept_id)) #Replace "RELATION_ID"

        # Action 2: No-op (or other actions, like removing concepts/relations)
        elif action_id == 2:
            pass  # Do nothing

        # Manage working memory (FIFO)
        while len(self.working_memory) > self.max_working_memory_size:
            self.working_memory.pop(0)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Store device for later use

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Concept embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings (optional)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.num_actions, bias=False) # Predict action IDs
       # self.transformer.wte.weight = self.lm_head.weight # No weight tying (usually)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The concept embeddings are *not* tied to the lm_head in this version.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, reasoning_state, concept_ids=None, targets=None): # Modified signature
        """
        Forward pass of the model.

        Args:
            reasoning_state: The current ReasoningState object.
            targets: Optional target action IDs for training.

        Returns:
            logits: Action logits (if targets is None) or None.
            loss: The loss (if targets is not None) or None.
        """
        # 1. Embed concepts from the reasoning state's working memory
        concept_ids_tensor = torch.tensor(reasoning_state.working_memory, dtype=torch.long, device=self.device)
        if concept_ids_tensor.numel() == 0:
          #  print("working memory empty") #for debugging
            concept_ids_tensor = torch.zeros((1,), dtype=torch.long, device=self.device) #Insert dummy, if working_memory is empty
        concept_embeddings = self.transformer.wte(concept_ids_tensor)

        # 2. (Optional) Add positional embeddings
        if self.transformer.wpe is not None:
          #  print("add positional embeddings")
            pos = torch.arange(0, concept_embeddings.size(0), dtype=torch.long, device=self.device) #changed size to first dimension
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(concept_embeddings + pos_emb)
        else:
            x = self.transformer.drop(concept_embeddings)


        # 3. Process with the Transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 4. Predict action logits (for the *last* element in the sequence)
        logits = self.lm_head(x[-1:, :])  # Only predict based on the last timestep


        # 5. Calculate loss (if targets are provided)
        loss = None
        if targets is not None:
           # print("Calculate loss")
            # Reshape logits to (batch_size * sequence_length, num_actions)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


        if targets is not None:
            return logits, loss
        else:
             # Inference: Choose the action with the highest probability
            probs = F.softmax(logits, dim=-1)
            action_id = torch.argmax(probs, dim=-1) # Get the most likely action ID
            return action_id, loss


    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # from_pretrained (Not needed for initial symbolic reasoning training)
       pass


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # This needs to be adapted for symbolic reasoning, as the original formula
        # is based on token processing.  It's likely that a closed-form solution
        # isn't possible, and you'll need to rely more on empirical measurements.
        # Placeholder for now:
        return 0.0


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        #Not used in this implementation.
       pass

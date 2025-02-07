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
    block_size: int = 1024  # Maximum context length (concepts)
    vocab_size: int = 50304  # Vocabulary size (number of concepts)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    relation_vocab_size: int = 0 #To be set later. Number of relations


class ReasoningState:
    def __init__(self, knowledge_graph, concepts=None, relations=None, working_memory=None, max_working_memory_size=10):
        self.concepts = concepts if concepts is not None else set()
        self.relations = relations if relations is not None else []
        self.working_memory = working_memory if working_memory is not None else []
        self.max_working_memory_size = max_working_memory_size
        self.knowledge_graph = knowledge_graph  # Store the KG

    def transition(self, concept_probs, relation_probs, concept_threshold=0.5, relation_threshold=0.5):
        """
        Updates the state based on predicted probabilities.

        Args:
            concept_probs: (vocab_size,) tensor of probabilities for each concept.
            relation_probs: (relation_vocab_size,) tensor of probabilities for each relation.
        """
        # 1. Update working memory based on concept probabilities
        predicted_concept_ids = (concept_probs > concept_threshold).nonzero(as_tuple=True)[0].tolist() # Get indices
        for concept_id in predicted_concept_ids:
            if concept_id not in self.working_memory:  # Avoid duplicates
                self.working_memory.append(concept_id)
                self.concepts.add(concept_id)


        # Keep working memory within size limits (FIFO)
        while len(self.working_memory) > self.max_working_memory_size:
            removed_concept = self.working_memory.pop(0)
             # Check if removed concept exists elsewhere, remove if not present.
            if removed_concept not in self.working_memory:
                self.concepts.discard(removed_concept)


        # 2. Update relations based on predicted probabilities and current working memory
        # Iterate through all *possible* pairs of concepts in working memory
        wm_tensor = torch.tensor(self.working_memory, dtype=torch.long, device=relation_probs.device)

        for i in range(len(self.working_memory)):
            for j in range(len(self.working_memory)):
                if i != j:  # Don't relate a concept to itself
                    concept_id1 = self.working_memory[i]
                    concept_id2 = self.working_memory[j]
                    # Check if predicted probability exceeds threshold
                    if relation_probs[i, j].mean() > relation_threshold:  # Use the mean probability

                        # Add relation (subject, predicate, object)
                        # We use itos and relation_itos to get the actual QID/PID strings
                        subj = self.knowledge_graph.itos_concepts[concept_id1] #Get QID
                        obj = self.knowledge_graph.itos_concepts[concept_id2]
                        rel = self.knowledge_graph.itos_relations[relation_probs[i,j].argmax().item()] # Get most likely relation

                        self.relations.append((subj, rel, obj)) #Append Tuple

        #Remove duplicate relations
        self.relations = list(set(self.relations))



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Positional embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output heads:
        self.concept_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Predict concept probabilities
        self.relation_head = nn.Linear(config.n_embd, config.relation_vocab_size, bias=False) #Predict relation probabilities


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

    def forward(self, reasoning_state, concept_targets=None, relation_targets=None):
        """
        Forward pass of the model.

        Args:
            reasoning_state: The current ReasoningState object.

            concept_targets: Optional (vocab_size,) tensor of target concept probabilities.
            relation_targets: Optional (relation_vocab_size,) tensor of target relation probabilities.

        Returns:
            concept_probs: (1, vocab_size) tensor of predicted concept probabilities.
            relation_probs: (1, relation_vocab_size) tensor of predicted relation probabilities.
            loss: The loss (if targets are provided) or None.
        """
        # 1. Embed concepts from the reasoning state's working memory
        concept_ids = torch.tensor(reasoning_state.working_memory, dtype=torch.long, device=self.device)

        # Handle the case where working memory is empty: important for training stability
        if concept_ids.numel() == 0:
            concept_ids = torch.zeros((1,), dtype=torch.long, device=self.device)  # Pad with a dummy concept ID

        concept_embeddings = self.transformer.wte(concept_ids)

        # 2. (Optional) Add positional embeddings
        if self.transformer.wpe is not None:
            pos = torch.arange(0, concept_embeddings.size(0), dtype=torch.long, device=self.device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(concept_embeddings + pos_emb) # Add here
        else:
             x = self.transformer.drop(concept_embeddings)

        # 3. Process with the Transformer
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (T, C)

        # 4. Predict concept and relation probabilities (using the *last* element in the sequence)

        concept_logits = self.concept_head(x[-1:, :])  # (1, vocab_size)
        relation_logits = self.relation_head(x) # (T, relation_vocab_size)

        # Reshape and pad relation_logits
        relation_logits = relation_logits.view(-1, self.config.relation_vocab_size)  # (T, relation_vocab_size)

        concept_probs = torch.sigmoid(concept_logits)  # (1, vocab_size)
        relation_probs = torch.sigmoid(relation_logits) # (T, relation_vocab_size)

        # 5. Calculate loss (if targets are provided)
        loss = None
        if concept_targets is not None and relation_targets is not None:
            # Use binary cross-entropy for independent concept/relation predictions
            concept_loss = F.binary_cross_entropy(concept_probs, concept_targets.float().unsqueeze(0))
            relation_loss = F.binary_cross_entropy(relation_probs, relation_targets.float()) #No unsqueeze, we use mean
            loss = concept_loss + relation_loss #Combine

        return concept_probs, relation_probs, loss

    def crop_block_size(self, block_size):
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
        # TODO: Adapt this for symbolic reasoning.
        return 0.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Basic building blocks
# ---------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm but with optional bias. PyTorch doesn't simply allow bias=False easily."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """
    Standard GPT-style multi-head masked attention.
    Expects `x` to have shape (B, T, C).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention')

        # For the older, manual masking path:
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        """
        x: (B, T, C)
        Returns: (B, T, C)
        """
        B, T, C = x.shape

        # project Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_size = C // self.n_head

        # reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            # PyTorch 2.0+ flash
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # manual masking
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # reassemble (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # final projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block: LN -> self-attn -> residual -> LN -> MLP -> residual
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.attn(self.ln_1(x))  # (B, T, C)
        x = x + self.mlp(self.ln_2(x))   # (B, T, C)
        return x


@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50000
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    relation_embd_dim: int = 64
    relation_vocab_size: int = 0


# Optional: single-sample state. Not used by multi-batch forward below.
class ReasoningState:
    def __init__(self, entity_itos, relation_itos,
                 concepts=None, relations=None, working_memory=None,
                 max_working_memory_size=10):
        self.entity_itos = entity_itos
        self.relation_itos = relation_itos
        self.concepts = concepts if concepts is not None else set()
        self.relations = relations if relations is not None else []
        self.working_memory = working_memory if working_memory is not None else []
        self.max_working_memory_size = max_working_memory_size

    def transition(self, concept_probs, relation_embeddings,
                   concept_threshold=0.5, relation_threshold=0.5):
        # user-defined logic
        pass


class GPT(nn.Module):
    """
    The multi-batch GPT that takes (B, T) concept IDs, returns (B, vocab_size) + (B, T, relation_embd_dim).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })

        # Heads
        self.concept_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.relation_embeddings = nn.Embedding(config.relation_vocab_size, config.relation_embd_dim)
        self.relation_final_layer = nn.Linear(config.n_embd, config.relation_embd_dim, bias=False)

        # Init weights
        self.apply(self._init_weights)
        # custom re-init on c_proj
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        concept_ids: torch.Tensor,    # (B, T)
        concept_targets: torch.Tensor = None,  # (B, vocab_size) for BCE
        relation_targets: torch.Tensor = None
    ):
        """
        concept_ids: (B, T)
        concept_targets: (B, vocab_size), e.g. one-hot vectors for each sample
        """
        B, T = concept_ids.shape
        assert T <= self.config.block_size, "Sequence length out-of-bounds."

        # 1) Embed => (B, T, n_embd)
        token_emb = self.transformer['wte'](concept_ids)

        # 2) Positional => (B, T, n_embd)
        positions = torch.arange(T, device=concept_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.transformer['wpe'](positions)  # (1, T, n_embd)
        pos_emb = pos_emb.expand(B, T, self.config.n_embd)

        # 3) Combine + dropout => (B, T, n_embd)
        x = self.transformer['drop'](token_emb + pos_emb)

        # 4) Pass through Transformer blocks => (B, T, n_embd)
        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)  # (B, T, n_embd)

        # 5) For concept classification, use the last token => (B, n_embd)
        x_last = x[:, -1, :]
        concept_logits = self.concept_head(x_last)  # (B, vocab_size)
        # If you want probabilities for logging, do:
        concept_probs = torch.sigmoid(concept_logits)  # (B, vocab_size)

        # 6) For relations, we project all tokens => (B, T, relation_embd_dim)
        relation_embed = self.relation_final_layer(x)

        # 7) Optional BCE loss with logits
        loss = None
        if concept_targets is not None:
            # safer to do with logits
            # so we do:
            # concept_logits shape => (B, vocab_size)
            # concept_targets shape => (B, vocab_size)
            loss = F.binary_cross_entropy_with_logits(
                concept_logits, concept_targets.float()
            )

            # if relation_targets is not None:
            #   # your relation loss, etc.

        return concept_probs, relation_embed, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # discount positional embeddings
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # crop positional embeddings
        self.transformer['wpe'].weight = nn.Parameter(
            self.transformer['wpe'].weight[:block_size]
        )
        # crop the attention mask in each block if needed
        for block in self.transformer['h']:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device_type == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # optional if measuring flops utilization
        return 0.0


    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        idx: (B, T) int tensor of concept IDs as prompt
        Returns: (B, T + max_new_tokens) with newly generated IDs appended
        """
        for _ in range(max_new_tokens):
            # Crop idx to last block_size tokens, in case it grows too large
            idx_cond = idx[:, -self.config.block_size:]  # (B, <=block_size)

            # Forward pass: we only need concept_probs from the last token
            concept_probs, _, _ = self(idx_cond)
            # concept_probs has shape (B, vocab_size)

            # The last forward pass returns the probabilities for the last token in each sequence
            # shape: (B, vocab_size)
            # We sample from these probabilities.
            logits = concept_probs / (temperature if temperature > 0 else 1.0)

            # Optionally top_k
            if (top_k is not None) and (top_k < logits.size(1)):
                # B x vocab_size
                v, ix = torch.topk(logits, top_k, dim=1)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, ix, v)
                logits = mask

            probs = F.softmax(logits, dim=-1)  # convert to probability
            next_id = torch.multinomial(probs, num_samples=1)  # shape (B, 1)

            # Append sampled ID to current sequence
            idx = torch.cat([idx, next_id], dim=1)  # (B, T+1)
        return idx


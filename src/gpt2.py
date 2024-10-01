from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#-------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        #projection of all k/v/q for all heads but in batch dimension
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        #output projection
        self.c_proj = nn.Linear(config.n_embed,config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size,config.block_size))
        
        
    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embed, dim=2)
        
        # (B,T,n_heads, C // n_head) => (B, n_heads, T, head_size)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_filled(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v #(B, n_heads, T, T) @ (B, n_heads, T, head_size)=> (B, n_heads, T, head_size)
        y = y.transpose(1,2).contiguous().view(B,T,C) #re-assemble; this is the concat operation
        #output projection 
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        
    def forwad(self, x):
        x = self.c_fc
        x = self.gelu
        x = self.c_proj
        return x 
        
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
            
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

@dataclass
class GPTConfig:
    block_size = 256
    vocab_size = 65
    n_layer=6
    n_head = 6
    n_embed = 384 
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict( #index using str
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.vocab_size, config.n_embed),
            h = nn.ModuleList(list( #index usinh int
                Block(config) for _ in range(config.n_layer)
                )),
            ln_f = nn.LayerNorm(config.n_embed)                                       
        )) 
        #projection from 756 to 507556
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        

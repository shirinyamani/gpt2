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
    block_size = 1024 # max_seq len 
    vocab_size = 50257
    n_layer=12
    n_head = 12
    n_embed = 768 
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
        
#------------loading GPT model weights from HF--------
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weight from {model_type}')
        
        config_args = {
            'gpt2': dict(n_layer= 12, n_head=12, n_embed= 768),
            'gpt2-medium': dict(n_layer= 24, n_head=16, n_embed= 1024),
            'gpt2-large':dict(n_layer= 36, n_head=20, n_embed= 1280),
            'gpt2-xl':dict(n_layer= 48, n_head=25, n_embed= 1600) 
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias')]
        
        #load the HF gpt2 model 
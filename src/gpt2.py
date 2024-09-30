from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#-------------------------------------
@dataclass
class GPTConfig:
    block_size = 256
    vocab_size = 65
    n_layer=6
    n_head = 6
    n_embed = 384
    
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
        

        
    
    
    
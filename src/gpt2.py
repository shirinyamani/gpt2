from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import os
#-------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        #projection of all k/v/q for all heads but in batch dimension
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        #output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.GPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size,config.block_size))
         
    def forward(self, x):
        B,T,C = x.size()
        #qkv = self.c_attn(x)
        q,k,v = self.c_attn(x).split(self.n_embed, dim=2)
        # (B,T,n_heads, C // n_head) => (B, n_heads, T, head_size)
        k = k.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C // self.n_head).transpose(1,2)
# manual implementation of attention

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # #att = self.attn_dropout(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs) 
        #flash attention, aware of the memory hierarchy
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.GPT_SCALE_INIT = 1
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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
    block_size: int = 1024 # max_seq len 
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768 
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict( #index using str
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList(list( #index usinh int
                Block(config) for _ in range(config.n_layer)
                )),
            ln_f = nn.LayerNorm(config.n_embed)                                       
        )) 
        #projection from 756 to 507556
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        #weight-sharing schema
        self.transformer.wte.weight = self.lm_head.weight
        
        # inint better wrt gpt2 paper
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)**-0.5 #scale down to compensate the std addition; 2 * bc every layer in out tf has 2 blocks that add the contribution; attn , mlp
            torch.nn.init.normal_(module.weight, mean=0, std=std) #the std 1/sprt(dimension); 1/sqrt(768)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)    
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
              
    def forward(self, idx, target=None):
        B ,T = idx.size() #(B,T)
        assert T <= self.config.block_size, f'cannot forward tensor size {T} to block size of {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T)
        pos_emb = self.transformer.wpe(pos) # (T, n_embed)
        tok_emb = self.transformer.wte(idx) # (B,T,n_embed)
        x = tok_emb + pos_emb # (B,T,n_embed)
        for block in self.transformer.h:
            x = block(x)  #(B,T,n_embed)
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # (B,T,vocab_size) for the token come next for 
        loss = None
        if target is not None:
            B,T,vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
           
#------------loading GPT model weights from HF--------
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f'loading weight from {model_type}')
        config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embed=768),
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),
        'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600)
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024 
        #config_args['bias'] = True 
        #print("forcing vocab_size=50257, block_size=1024, bias=True")
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert sd_keys == sd_keys_hf
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # print(f'shape hf model keys:{sd_hf[k].shape}, our model key shape is {sd[k].shape}')
                # assert model.transformer.wte.weight.shape == model.lm_head.weight.shape, "Mismatch between wte and lm_head shapes."
                # # vanilla copy over the other parameters
                if sd_hf[k].shape != sd[k].shape:
                    print(f"Problematic key: {k}")
                assert sd_hf[k].shape  == sd[k].shape, f'mismatched {sd_hf[k].shape} != {sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model #return gpt object
    
    def configure_optimizer(self, weight_decay, lr, device):
        #1) retreive the params that require grad
        model_params_dict = {pn:p for pn,p in self.named_parameters()}
        model_params_dict = {pn:p for pn,p in model_params_dict.items() if p.requires_grad}
        
        #2) devide the params by the ones to decay or not to decay based on the dim
        param_decay = [p for n,p in model_params_dict.items() if p.dim() >= 2] #like weights and embeddings
        param_no_decay = [p for n,p in model_params_dict.items() if p.dim() < 2]
        
        #3) define them in the gp
        optim_groups = [
            {'params': param_decay, 'weight_decay': weight_decay},
            {'params':param_no_decay,'weight_decay': 0.0 }
        ]
        
        #how many of params are we decaying
        num_decay = sum(p.numel() for p in param_decay) #ya wanna w decay mostly the w in the multipications and in the embedding n not allowing them to be individually too large!
        num_no_decay = sum(p.numel() for p in param_no_decay) #like biases, scales
        print(f'num decayed params:{num_decay} with dim >= 2')
        print(f'num decayed params:{num_no_decay} with dim < 2')
        
        #create adamw optim n check if fuzed of it is avail
        fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_avail and 'cuda' in device 
        print(f'used_fused: {used_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

#============DEVICE========================
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

#==============TOKENIZATION & DataLoader==================
#get a batch 
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        #enumerate on the shards to read the data
        data_root = "../data/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = sorted(list(s for s in shards if split in s))
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, "No shards found for {split} split"
        #read data from shards
        self.current_shard = 0 #start from the first shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T
        self.reset()
        
    def reset(self):
        B, T = self.B, self.T 
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = B * T  
        
    def next_batch(self):
        B, T = self.B, self.T 
        buf = self.tokens[self.current_position: self.current_position + B*T+1]
        x = buf[:-1].view(self.B, self.T) #input
        y = buf[1:].view(self.B, self.T) #target
        self.current_position += B * T
        #when we run out of token in a single shard
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y
    

#==============LOAD MODEL=================
#model = GPT.from_pretrained(model_type='gpt2')
import time 
model = GPT(GPTConfig(vocab_size=50304)) #to be nice num
#print(f'Successfully loaded the weights from {model._get_name()}')
model.to(device)
model = torch.compile(model)
print(f'using device: {device}')

#lr function according to the gpt3 paper
max_lr = 6e-4
min_lr = max_lr* 0.1 #10% of the above according to paper 
warmup_steps = 100
max_steps = 2000

def get_lr(it):
    if it < warmup_steps:
        return max_lr* (it+1) / warmup_steps #+1 bc we dont wanna start by 0 lr
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)


#reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

 #gradient_accumulation: simulate any arbitary batch size with serializing the gradient 
ultimate_batch_size = 524288 #2**19
B=64
T=1024
assert ultimate_batch_size % (B * T) == 0, "make sure ultimate batch size is dividable by the B*T"
grad_accum_steps = ultimate_batch_size // (B * T) #forward/backword and all the grad will be += untill we touch the ultimate then single update
print(f'accum steps serialized: {grad_accum_steps} for the ultimate batch size {ultimate_batch_size}')

#LOAD DATA IN BATCHES
train_loader = DataLoaderLite(B=B, T=T, split='train')
val_loader = DataLoaderLite(B=B, T=T, split='val')
#mixed precision
torch.set_float32_matmul_precision('high') 

#Optimize!
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, lr=max_lr, device=device)

#For inference/validation 
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps -1)
    
    #Once in a while (every 100 steps) evaluate the model
    if (step > 0 and step % 100 == 0) or last_step: #once in a while eval the model 
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision training 
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            print(f'Step {step} | loss: {val_loss_accum.item():.4f}')
    
    #Once in a while generate from the model 
    if (step > 0 and step % 100 == 0) or last_step:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens= enc.encode("Hello, I'm a language model, and I")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        #GENERATE w /Prefix Token
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) #(B,T, vocab_size)
                logits = logits[:,-1,:] #(B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                #according to hf get only top 50 high probs
                top_kprob, top_kindic = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(top_kprob, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(top_kindic, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            words = enc.decode(tokens)
            print(">", words)

#for Training
    model.train()
    optimizer.zero_grad()
    #grad accum
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): #mixed precision training 
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #set lr t the function
    lr = get_lr(step)
    #set lr in pytorch
    for gp_params in optimizer.param_groups:
        gp_params['lr'] = lr
        
    optimizer.step()
    torch.cuda.synchronize() #to force the queue; waiting fir the gpu to fiish the started job
    t1 = time.time()
    throghput = (train_loader.B * train_loader.T * grad_accum_steps) / (t1-t0) #howmany tokens per second we're processing
    print(f'Step {step} | loss: {loss_accum.item():.4f} | total time: {(t1-t0)*1000:.2f} | norm:{norm:.3f} | lr:{lr:.4e}|  total tok/sec: {throghput:.2f}') #float on cpu

import sys; sys.exit(0)

#TODO:  1) torch.gather(), 2) investigate shape of the idx, ix 

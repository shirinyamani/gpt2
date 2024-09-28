#Baseline: bigram 
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions? 256 tokens get in to predict 257th
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384 #dimension of each of the words for the nn.Embedding lookup 
n_head = 6 #384/6 = 64 every head takes 64 dimensional
n_layer = 6
dropout = 0.2 #everu forward/backward pass 20% of the intermideiate calculations are disabled n droppend to zero
torch.manual_seed(1337)

#Get the data 
with open(file='./data/input.txt', mode='r',  encoding='utf-8') as f:
    text = f.read()


#data
chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenization 
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] 
decode = lambda numz: ''.join([itos[i] for i in numz])
    

#tokenization on entire dataset   
data = torch.tensor(encode(text), dtype= torch.long)
#split to train/test
n = int(0.9 * len(data))
train_set = data[:n]
test_set = data[n:]


#data_loader 
#time dimension
def get_batch(split):
    data = train_set if split == "train" else test_set
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False) #apply matrix mul
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x) # B, T, C
        q = self.query(x) #  B, T, C
        
        wei = q @ k.transpose(-2,-1) * C**0.5 # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T]== 0, float('-inf')) # B,T,T
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) #b,t,c
        out = wei @ v # (B,T,T) @ (B,T,C) ==> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1) #concat over channel dimension
        out = self.proj(out) #projectiion is the linear transformation of the out layer
        return out #projection back to the residual pathway
    
    
class FeedForward(nn.Module):
    """simple linear layer y = wx + b followed by RelU"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 4 * is based on the paper: growing the layer on the side on res pathway
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), #projection layer back to the residual pathway 
            nn.Dropout(dropout) 
            )
    def forward(self, x):
        return self.net(x)

class  Block(nn.Module):
    """TF block: interperse communication (attention) and computation"""  
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head   
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size) #communication #32 /4 ==> each head 8 in parallel
        self.ffwd = FeedForward(n_embed) #computation
        self.ln1 = nn.LayerNorm(n_embed) #normalize features guassian 0 mean, 1 std
        self.ln2 = nn.LayerNorm(n_embed)


    def forward(self, x):
        x = x + self.sa(self.ln1(x)) #residual connection; we forked off, did some computation and come back to x 
        x = x + self.ffwd(self.ln2(x))
        return x              
#Baseline: bigram 
#Bi-gram model 
class BigramLanguageModel(nn.Module):
    def __init__(self):
        """ nn.Embedding is a wrapper
        input: list of indices.
        out: corresponding word embeddings
        """
        super().__init__()
        #wrapper 
        self.token_embeding_table = nn.Embedding(vocab_size, n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #each position from 0 to blocksize-1
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embed) #final layer norm befor the lang model head 
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):
        B ,T = idx.shape
        token_embed = self.token_embeding_table(idx) #(B,T, C)
        positional_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)thro the embed table
        x = token_embed + positional_embed #(B,T, C) broadcasted across Batch 
        x = self.blocks(x) #(B,T,C)
        x = self.ln_final(x) # (B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)
        #print(logits.shape)
        if target is None:
            loss = None 
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #streched 
            # print(f'logits shape is {logits.shape} and it looks like:')
            # print(logits)
            target = target.view(B*T) #-1 streched 
            # print(f'logits shape is {target.shape} and it looks like:')
            # print(target)
            loss = F.cross_entropy(logits, target) #should be (B*T,C)  # LOSS: -ln(1/65)
            #print(f'idx shape is {idx.shape}')
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        #idx shape is (B, T) which is the current context within some batch
        #so the job of generate is to take the (B,T) and extend it to (B, T + 1) in all the time dimensions within the batches
        for _ in range(max_new_tokens):
            #get predictions
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] #becomes (B, C)
            probs = F.softmax(logits, dim=-1) #(B, C)
            #sample from this probs
            # If input is a matrix with m rows, out is an matrix of shape (m * num_samples)
            idx_next = torch.multinomial(probs, num_samples=1) # (B * 1) 
            #appending the sample to the current seq
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx 
    
model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
    # get a sample batch 
    x_batch, y_b = get_batch(split='train')
    #forward/backward evaluate the loss
    logits, loss = model( x_batch, y_b)
    #zero out all the gradients from prev steps 
    optimizer.zero_grad(set_to_none=True)
    #gettting the gradients for all the params 
    loss.backward()
    #using those gradients to update our params
    optimizer.step()
    
    
#genration 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=700)[0].tolist()))
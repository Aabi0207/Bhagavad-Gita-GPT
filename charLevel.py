# import the Requirements
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import tqdm

# Use below setting only on gpu

# Constants
batch_size = 64
n_embd = 384
n_head = 6
eval_iters = 100
block_size = 128
n_layer = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 500
max_iter = 5000
dropout = 0.3
lr = 3e-4

# Load the data
data_dir = "/kaggle/input/bhagavad-gita-encoded"
def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ixs = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ixs])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int32)) for i in ixs])

    x, y = x.to(device), y.to(device)
    return x.long(), y.long()

with open(os.path.join(data_dir, 'meta.pkl'), "rb") as f:
    meta = pickle.load(f)

vocab_size = meta.get("vocab_size")
itos = meta.get("itos")
decode = lambda l: "".join([itos[id] for id in l])

# Estimate the function to evaluate the model
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value =nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):   # B, T, C
        B,T,C = x.shape
        k = self.key(x)  # B, T, hs
        q = self.query(x)# B, T, hs
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # B, T, HS @ B, HS, T == B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) #B, T, HS
        out = wei @ v # B,T, HS
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # B,T,HS*N_HEAD
        out = self.proj(out) # B,T,C
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  #B,T,C


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x   # B,T,C


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lif = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, model):
        if isinstance(model, nn.Linear):
            torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)
            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)
        if isinstance(model, nn.Embedding):
            torch.nn.init.normal_(model.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lif(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_token):
        for _ in range(max_token):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


model = GPTLanguageModel().to(device)

print(sum(p.numel() for p in model.parameters()))

opt = torch.optim.AdamW(model.parameters(), lr=lr)

best_val_loss = float('inf')

for i in tqdm.tqdm(range(max_iter)):
    # forward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    # backward pass
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    # every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0 or i == max_iter - 1:
        losses = estimate_loss()

        # Save the model with the best validation loss
        if losses['val'] < best_val_loss:
            torch.save(model.state_dict, 'model.pth')
        print(f"step {i}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")


# load the saved model
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load("model.pth"))

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
open('more.txt', 'w').write(decode(model.generate(context, max_token=10000)[0].tolist()))
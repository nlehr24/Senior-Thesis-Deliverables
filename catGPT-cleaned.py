import time
import os
import numpy as np
import librosa
import wave
start = time.time()
rootdir = '/Users/nlehr24/Downloads/dataset'    # downloaded from https://zenodo.org/records/4008297
meowPath = '/Users/nlehr24/Downloads/440Meows.wav'

sigFigs = 3    # number found from trial and error, not certain why others work so differently

def writeMeowFile():    # no need to run this after written once?
    meows = []
    for subdir, dirs, files in os.walk(rootdir):
        meows = files

    bigMeowWav = []

    for path in meows:
        data, sampleRate = librosa.load(rootdir + '/' + path, sr=None)  # sr must equal None to preserve the samplerate of the original file, else defaults to 22050
        bigMeowWav += [round(d, sigFigs) for d in data] # rounding when writing is useful despite the scaling up later; 
                                                        #various testing has indicated that things just work more smoothly in general when truncating data early on
    assert sampleRate == 8000, 'unexpected sampleRate for bigMeowWav'

    import soundfile as sf
    sf.write(meowPath, bigMeowWav, sampleRate) 

    print('Time taken to write:', round(time.time() - start, 2), 'seconds')

# writeMeowFile()

def visualize_audio(y, sr, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, color='blue')    # color specification necessary, else prop_cycler error nonsense
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 400
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1234)

data, sampleRate = librosa.load(meowPath, sr=None)   # extracting data and sample rate from 440Meows.wav via librosa

visualize_audio(data, sampleRate, 'Full 440 Meow Dataset')
visualize_audio(data[3000:11000], sampleRate, '1s of 440 Meow Dataset')

scale = 10 ** (sigFigs)
print('Number of data points:', len(data))
print('Data maximum:', max(data))
print('Data minimum:', min(data))

MIN = min(data)
# scaling up to get int (0.1234 * 10^4 == 1234.0) & adding min (subtracting negative) so that there are no negative index problems
data = np.array([(d - MIN) * scale for d in data]) 

print('Data maximum after scaling:', max(data))
print('Data minimum after scaling:', min(data))


vocab_size = len(set(list(data)))   # vocab_size comes before conversion to tensor because set() doesn't eliminate tensor dupes

print('Number of unique inputs:', vocab_size)

stragglers = [d for d in data if d < 0 or d >= vocab_size]
print('Number of cut-out inputs:', len(stragglers))

print(f'Percentage of inputs cut out: {round( ( len(stragglers) / len(data) ) * 100, 3)}%')
data = np.array([d for d in data if 0 <= d < vocab_size])  # forcefully eliminating all inputs that are too big or negative while chopping off decimal
data = torch.tensor(data, dtype=torch.long) 

n = int(0.9*len(data))  # splitting 90% for training data and 10% for testing
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split, block_size=block_size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y     # both with shapes of (batch_size, block_size)

# model classes entirely from Andrej Karpathy - https://github.com/karpathy/ng-video-lecture 
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape   # batch, time, channels
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer; AdamW used because "Adam is better than other adaptive learning rate algorithms due to its faster convergence and robustness across problems" - https://builtin.com/machine-learning/adam-optimization 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step() 

# generate from the model, from an input of (0, 0)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
outp = m.generate(context, max_new_tokens=12000)[0].tolist()     #vwith a sampleRate of 8000, time of clip produced = max_new_tokens/8000

outp = np.array([(d / scale) + MIN for d in outp])

visualize_audio(outp, sampleRate, f'1s of Generated Meowing (After {max_iters} Training Iterations)')

def play():
    newFileName = f'/Users/nlehr24/Downloads/Generated-Meowing-(After-{max_iters}-Training-Iterations.wav'
    print('Playing sound...')

    import soundfile as sf
    sf.write(newFileName, outp, sampleRate)
    
    from playsound import playsound
    playsound(newFileName)

    
play()


print(f'Time taken: {round(time.time() - start, 2)} seconds')
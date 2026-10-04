import os
import urllib.request
import torch
import torch.nn as nn
import ssl
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
head_size = 16

torch.manual_seed(1337)

# GÖREV 1

if not os.path.exists('input.txt'):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=context) as response, open('input.txt', 'wb') as out_file:
        out_file.write(response.read())
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

bigram_model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(bigram_model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = bigram_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

base_losses = estimate_loss(bigram_model)
print(f"--> Bigram Sonuçları: Train Loss: {base_losses['train']:.4f}, Val Loss: {base_losses['val']:.4f}\n")


# GÖREV 2

# Matris çarpımı neden ağırlıklı ortalama?
# wei@x işleminde t. satırı, x'in satırlarının wei[t] katsayılarıyla toplanmasıdır.
# tril yönteminde wei[t] satırı ilk t+1 elemanda eşit (1/(t+1)), geri kalanda 0'dır.
# Yani ağırlıklar negatif değil ve toplamı 1, bu da ağırlıklı ortalamanın tanımı.


torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# For döngüsü
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

# torch.tril
wei2 = torch.tril(torch.ones(T, T))
wei2 = wei2 / wei2.sum(1, keepdim=True)
xbow2 = wei2 @ x

# Softmax
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
wei3 = wei3.masked_fill(tril == 0, float('-inf'))
wei3 = F.softmax(wei3, dim=-1)
xbow3 = wei3 @ x

print((xbow - xbow2).abs().max())
print((xbow - xbow3).abs().max())
print(xbow.shape, xbow2.shape, xbow3.shape)
print("for torch.tril'e eşit mi?", torch.allclose(xbow, xbow2, atol=1e-6))
print("for softmax'e eşit mi ?:", torch.allclose(xbow, xbow3, atol=1e-6))


# GÖREV 3
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)
head_size = 16

# Neden head_size?
# q ve k'nın elemanları birim varyanslıysa, head_size boyutlu iki vektörün skalar çarpımı head_size tane çarpımın toplamıdır.
# Varyansı yaklaşık head_size olur, standart sapması head_size.

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B, T, head_size)
q = query(x) # (B, T, head_size)

# Scaling işlemi ve wei hesabı
wei_raw = q @ k.transpose(-2, -1) 
wei_scaled = wei_raw * (head_size ** -0.5)

tril = torch.tril(torch.ones(T, T))
wei_masked = wei_scaled.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei_masked, dim=-1)

v = value(x)
out = wei @ v

print("Örnek 0 için Ağırlık Matrisi (wei[0]):")
print(wei[0])
print("\nMatrisin 4. Satırının İncelemesi:")
print(wei[0, 3].detach().numpy())

"""
[GÖREV 3 AÇIKLAMA / YORUM - Satır İncelemesi]:
wei[0, 3] satırı, 4. sıradaki token'ın kendisinden önceki 1, 2, 3 ve 4. token'lara ne kadar 'dikkat' (attention)
ayırdığını gösterir. Maskeleme sebebiyle 5, 6, 7 ve 8. token'ların ağırlığı tam olarak 0.0000'dır (geleceğe bakamaz).
Satırdaki olasılık değerlerinin toplamı 1.0'dir ve en yüksek katsayıya sahip olan token, 4. token'ın en çok bilgi
topladığı geçmiş karakteri ifade eder.
"""


q_demo = torch.randn(B, T, head_size) * 3
k_demo = torch.randn(B, T, head_size) * 3
wei_demo_raw = q_demo @ k_demo.transpose(-2, -1)

softmax_unscaled = F.softmax(wei_demo_raw[0, 0], dim=-1)
softmax_scaled = F.softmax(wei_demo_raw[0, 0] / (head_size**0.5), dim=-1)

print("\nScaling (Bölme) Etkisi ")
print("Bölünmemiş Softmax:")
print(softmax_unscaled.detach().numpy())
print("Bölünmüş Softmax:")
print(softmax_scaled.detach().numpy())






# GÖREV 4
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

class SelfAttentionBigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.sa_head(x) 
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

sa_model = SelfAttentionBigramModel().to(device)
optimizer = torch.optim.AdamW(sa_model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(sa_model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = sa_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

sa_losses = estimate_loss(sa_model)
print("Karşılaştırma ve Sonuç")
print(f"Taban Bigram Modeli -> Val Loss: {base_losses['val']:.4f}")
print(f"Self-Attention'lı Model -> Val Loss: {sa_losses['val']:.4f}")

# Val loss neden düştü?
# Bigram her harfi yalnızca bir önceki harfe bakarak tahmin eder. 
# Self-attention'lı model önceki block_size harfe bakabilir ve hangisinin önemli olduğunu öğrenir.
# Bu düşüş küçük de olabilir çünkü kapasite sınırlı.


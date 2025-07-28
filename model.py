import math
import yaml
import numpy as np
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@dataclass
class BERT_Config:
    config = load_config()

class BERT_Embedding(nn.Module):
    def __init__(self, vocab_size: int, num_embd: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, num_embd)
        self.positional_embeddings = nn.Embedding(vocab_size, num_embd)
        self.layer_norm = nn.LayerNorm(dims=num_embd)

    def __call__(self, tokens: mx.array):
        positions = mx.arange(stop=tokens.shape(1))
        tokens = self.token_embeddings(tokens) + self.positional_embeddings(positions)
        return self.layer_norm(tokens)
    
class BERT_Attention(nn.Module):
    def __init__(self, num_embd: int, num_heads: int):
        super().__init__()
        self.num_embd = num_embd
        self.num_heads = num_heads

        assert num_embd % num_heads == 0
        self.head_dim = num_embd // num_heads
        self.qkv = nn.Linear(num_embd, num_embd*3) # we multiply by 3 to get the q, k, and v matrices together
        self.output = nn.Linear(num_embd, num_embd)

    def __call__(self, tokens: mx.array, mask):
        B, T, C = tokens.shape() # B = Batch size, T = Block size, C = Channel Size
        qkv = self.qkv(tokens).reshape(B, T, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attn_probs = attention.softmax(dim=-1)
        context = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        return self.output(context)

class Block(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.attention = BERT_Attention(config.num_embd, config.num_heads)
        self.layer_norm1 = nn.LayerNorm(dims=config.num_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(config.num_embd, config.hidden_embd),
            nn.GELU(),
            nn.Linear(config.hidden_embd, config.num_embd)
        )
        self.layer_norm2 = nn.LayerNorm(dims=config.num_embd)

    def __call__(self, tokens: mx.array, mask=None):
        tokens = self.layer_norm1(tokens + self.attention(tokens, mask))
        tokens = self.layer_norm2(tokens + self.ff(tokens))
        return tokens

class BERT_Classification(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()

    def __call__(self, tokens: mx.array):
    

class BERT(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config

        self.transformer = {
            "embedding": BERT_Embedding(config.vocab_size, config.num_embd),
            "blocks": [Block(config) for _ in range(config.num_layers)],
            "output_proj": BERT_Classification(config.num_embd, config.vocab_size)
        }

    def __call__(self, tokens: mx.array):
        x = self.transformer[embedding](x)
        for block in self.transformer[blocks]:
            x = block(x)
        return self.transformer[output_proj](x) 

if __name__ == "__main__":
    print("hello")
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
    num_embd: int
    num_layers: int
    num_heads: int
    hidden_embd: int
    vocab_size: int
    block_size: int

    @staticmethod
    def from_yaml(path="config.yaml"):
        config = load_config(path)
        return BERT_Config(
            num_embd=config['model']['num_embd'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            hidden_embd=config['model']['hidden_embd'],
            vocab_size=config['model']['vocab_size'],
            block_size=config['model']['block_size']
        )

class BERT_Embedding(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, num_embd: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, num_embd)
        self.positional_embeddings = nn.Embedding(block_size, num_embd)
        self.layer_norm = nn.LayerNorm(dims=num_embd)

    def __call__(self, tokens):
        B, T = tokens.shape

        tokens = mx.array(tokens, dtype=mx.int32)
        positions = mx.arange(T)
        positions = mx.broadcast_to(positions, (B, T))
        positions = mx.array(positions, dtype=mx.int32)

        tokens_emb = self.token_embeddings(tokens)
        positions_emb = self.positional_embeddings(positions)

        return self.layer_norm(tokens_emb + positions_emb)
    
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
        B, T, C = tokens.shape
        qkv = self.qkv(tokens).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (0, 3, 1, 2, 4))
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        attention = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)

        if mask is not None:
            attention = mx.where(mask == 0, -1e9, attention)

        attn_probs = mx.softmax(attention, axis=-1)
        context = attn_probs @ v
        context = mx.transpose(context, (0, 2, 1, 3)).reshape(B, T, C)

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
        tokens = self.layer_norm2(tokens + self.feedforward(tokens))
        return tokens

class BERT(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config
        self.embedding = BERT_Embedding(config.vocab_size, config.block_size, config.num_embd)
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        self.output_proj = nn.Linear(config.num_embd, config.vocab_size)

    def __call__(self, tokens: mx.array):
        tokens = self.embedding(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_proj(tokens) 

if __name__ == "__main__":
    config = BERT_Config.from_yaml()
    model = BERT(config)
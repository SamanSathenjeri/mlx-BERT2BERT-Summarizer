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
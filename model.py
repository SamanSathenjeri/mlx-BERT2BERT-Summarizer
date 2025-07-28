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

class BERT_Embedding():


class BERT(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config

        self.transformer = {
            "embedding": BERT_Embedding(config.vocab_size, config.num_embd),
            "blocks": [Block(i, config) for i in range(config.num_layers)],
            "output_proj": BERT_Classification(config.num_embd, config.vocab_size)
        }

    def forward(self, tokens: mx.array):
        x = self.transformer[embedding](x)
        for block in self.transformer[blocks]:
            x = block(x)
        return self.transformer[output_proj](x) 

if __name__ == "__main__":
    print("hello")
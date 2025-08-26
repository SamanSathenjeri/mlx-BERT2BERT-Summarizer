import math
import yaml
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

def load_config(config_path="config.yaml"):
    '''
    Function to load config.yaml file
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@dataclass
class BERT_Config:
    '''
    BERT model config
    
    Params:
    - num_embd (int): number of embedding dimensions
    - num_layers (int): number of transformer blocks
    - num_heads (int): number of attention heads per multiheaded attention layer
    - hidden_embd (int): Feedforward dimension size (num_embd * 4)
    - vocab_size (int): Number of tokens in the dictionary
    - block_size (int): Context size
    - dropout (float): Percentage of layer nodes to drop out of training
    '''
    num_embd: int
    num_layers: int
    num_heads: int
    hidden_embd: int
    vocab_size: int
    block_size: int
    dropout: float

    '''
    Function to load from config file and set local config variables
    '''
    @staticmethod
    def from_yaml(path="config.yaml"):
        config = load_config(path)
        return BERT_Config(
            num_embd=config['model']['num_embd'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            hidden_embd=config['model']['hidden_embd'],
            vocab_size=config['model']['vocab_size'],
            block_size=config['model']['block_size'],
            dropout=config['model']['dropout'],
        )

class BERT_Embedding(nn.Module):
    '''
    BERT embedding layer implementation
    '''
    def __init__(self, config: BERT_Config):
        '''
        Initializing the weight matrices and positional embeddings
        
        Params:
        config (BERT_Config) - config instance holding the model hyperparameters
        '''
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.num_embd)
        self.positional_embeddings = nn.Embedding(config.block_size, config.num_embd)

        # # creating positional embeddings
        # positional_embeddings = mx.zeros([config.block_size, config.num_embd], dtype=mx.float32)
        # position = mx.arange(0, config.block_size, dtype=mx.float32)[:, None]
        # div_term = mx.exp(mx.arange(0, config.num_embd, 2, dtype=mx.float32) * -(math.log(10000.0) / config.num_embd))

        # positional_embeddings[:, 0::2] = mx.sin(position * div_term)
        # positional_embeddings[:, 1::2] = mx.cos(position * div_term)
        # self.positional_embeddings = positional_embeddings[None]

        self.segmenting = nn.Embedding(3, config.num_embd)
        self.layer_norm = nn.LayerNorm(dims=config.num_embd)
        self.dropout = nn.Dropout(config.dropout)

        # print(self.token_embeddings.items())
        # print(self.positional_embeddings)

    def __call__(self, tokens):
        B, T = tokens.shape
        tokens = tokens.astype(mx.int32) 

        # making a positions array, to get the embeddings for them
        positions = mx.arange(T)
        positions = mx.broadcast_to(positions, (B, T))
        positions = positions.astype(mx.int32) 

        tokens_emb = self.token_embeddings(tokens)
        positions_emb = self.positional_embeddings(positions)
        # positions_emb = self.positional_embeddings[:, :positions]
        # segmenting_emb = self.segmenting(tokens)

        # x = self.layer_norm(tokens_emb + positions_emb + segmenting_emb)
        x = self.layer_norm(tokens_emb + positions_emb)
        return self.dropout(x)

class BERT_Attention(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.num_heads = config.num_heads
        assert config.num_embd % self.num_heads == 0
        self.head_dim = config.num_embd // self.num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(config.num_embd, config.num_embd)
        self.k = nn.Linear(config.num_embd, config.num_embd)
        self.v = nn.Linear(config.num_embd, config.num_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.num_embd, config.num_embd)

    def __call__(self, x, mask=None, return_attn=False):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = self.k(x).reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = self.v(x).reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        attn_scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / self.scale  # (B, H, T, T)
        if mask is not None:
            if mask.ndim == 2:  # (B, T)
                mask = mask[:, None, None, :]  # -> (B, 1, 1, T)
            attn_scores = mx.where(mask == 0, -1e9, attn_scores)

        attn_weights = self.dropout(mx.softmax(attn_scores, axis=-1))
        output = attn_weights @ v
        output = output.transpose((0, 2, 1, 3)).reshape(B, T, -1)

        if return_attn:
            return self.output(output), attn_weights
        return self.output(output)

class Block(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.attention = BERT_Attention(config)
        self.layer_norm1 = nn.LayerNorm(dims=config.num_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(config.num_embd, config.hidden_embd),
            nn.GELU(),
            nn.Linear(config.hidden_embd, config.num_embd)
        )
        self.dropout = nn.Dropout(config.dropout)

        self.layer_norm2 = nn.LayerNorm(dims=config.num_embd)

    def __call__(self, tokens: mx.array, mask=None):
        tokens = tokens + self.dropout(self.attention(self.layer_norm1(tokens), mask))
        tokens = tokens + self.dropout(self.feedforward(self.layer_norm2(tokens)))
        return tokens

class BERT(nn.Module):
    '''
    Main class for the BERT model

    config (BERT_Config) - config instance holding the model hyperparameters
    embedding (BERT_Embedding) - reference to class creating and computing the embeddings
    blocks (List<Block>) - list of block objects for model to train through
    encoder_Flag (bool) - flag to return logits (for normal training) or computed tensor (for cross attention)
    output_proj (nn.Linear) - linear layer to project output tensor to token predictions (logits)
    '''
    def __init__(self, config: BERT_Config, encoder_Flag=False):
        '''
        Initialization module for the BERT model
        
        Params:
        config (BERT_Config) - config instance holding the model hyperparameters
        encoder_Flag (bool) - flag to return logits (for normal training) or computed tensor (for cross attention)
        '''
        super().__init__()
        self.config = config
        self.embedding = BERT_Embedding(config)

        # self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        self.encoder_Flag = encoder_Flag
        self.output_proj = nn.Linear(config.num_embd, config.vocab_size)

    def __call__(self, tokens: mx.array, mask=None, return_embeddings=False):
        '''
        Forward pass for the model
        
        Params:
        tokens (mx.array) - incoming tokens from the training or generation set
        mask (mx.array) - 
        return_embeddings (bool) - 
        '''
        tokens = self.embedding(tokens)
        assert not mx.any(mx.isnan(tokens)).item(), "NaNs in embedding"

        # tokens = self.blocks(tokens)
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, mask=mask)

        if not self.encoder_Flag:
            return self.output_proj(tokens)
        return tokens

if __name__ == "__main__":
    config = BERT_Config.from_yaml()
    model = BERT(config)
    mx.eval(model.parameters())
    # with open("/backup/open.txt", "w") as file:
    #     file.write(str(model.parameters()))
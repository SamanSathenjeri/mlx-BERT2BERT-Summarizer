import math
import mlx.core as mx
import mlx.nn as nn

from BERTEncoder import BERT, BERT_Config, BERT_Attention, BERT_Embedding

def causal_mask(seq_len: int):
    return mx.tril(mx.ones((seq_len, seq_len)))

def make_padding_mask(input_ids: mx.array, pad_token_id: int = 0):
    return (input_ids != pad_token_id)[..., None, None, :]

def make_decoder_self_mask(input_ids, pad_token_id=0):
    seq_len = input_ids.shape[1]
    causal = causal_mask(seq_len)[None, None, :, :]
    padding = make_padding_mask(input_ids, pad_token_id=pad_token_id)
    return causal & padding

class CrossAttention(nn.Module):
    def __init__(self, num_embd, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = num_embd // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q = nn.Linear(num_embd, num_embd)
        self.k = nn.Linear(num_embd, num_embd)
        self.v = nn.Linear(num_embd, num_embd)
        self.output = nn.Linear(num_embd, num_embd)

    def __call__(self, x, context, mask=None):
        B, T, _ = x.shape

        q = self.q(x).reshape(B, T, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = self.k(context).reshape(B, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = self.v(context).reshape(B, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))

        attn_scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / self.scale

        if mask is not None:
            attn_scores = mx.where(mask == 0, -1e9, attn_scores)

        attn_weights = mx.softmax(attn_scores, axis=-1)
        output = attn_weights @ v
        output = output.transpose((0, 2, 1, 3)).reshape(B, T, -1)

        return self.output(output)

class DecoderBlock(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.self_attn = BERT_Attention(config.num_embd, config.num_heads)
        self.cross_attn = CrossAttention(config.num_embd, config.num_heads)
        self.ff = nn.Sequential(
            nn.Linear(config.num_embd, config.hidden_embd),
            nn.GELU(),
            nn.Linear(config.hidden_embd, config.num_embd)
        )

        self.layer_norm1 = nn.LayerNorm(dims=config.num_embd)
        self.layer_norm2 = nn.LayerNorm(dims=config.num_embd)
        self.layer_norm3 = nn.LayerNorm(dims=config.num_embd)

    def __call__(self, tokens, encoder_hidden_states, self_mask=None, cross_mask=None):
        tokens = self.layer_norm1(tokens + self.self_attn(tokens, mask=self_mask))
        tokens = self.layer_norm2(tokens + self.cross_attn(tokens, encoder_hidden_states, mask=cross_mask))
        tokens = self.layer_norm3(tokens + self.ff(tokens))
        return tokens

class BERTDecoder(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config
        self.embedding = BERT_Embedding(config.vocab_size, config.block_size, config.num_embd)
        self.blocks = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.output_proj = nn.Linear(config.num_embd, config.vocab_size)

    def __call__(self, tokens: mx.array, encoder_hidden_states, decoder_pad_mask, encoder_pad_mask):
        self_mask = make_decoder_self_mask(tokens)
        cross_mask = encoder_pad_mask

        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, encoder_hidden_states, self_mask=self_mask, cross_mask=cross_mask)
        return self.output_proj(x)

class BERT2BERT(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.encoder = BERT(config)
        self.decoder = BERTDecoder(config)
        self.lm_head = nn.Linear(config.num_embd, config.vocab_size, bias=False)

    def __call__(self, encoder_input_ids, decoder_input_ids):
        encoder_pad_mask = make_padding_mask(encoder_input_ids)
        decoder_pad_mask = make_padding_mask(decoder_input_ids)

        encoder_output = self.encoder(encoder_input_ids, mask=encoder_pad_mask)
        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_output,
            decoder_pad_mask=decoder_pad_mask,
            encoder_pad_mask=encoder_pad_mask
        )
        return self.lm_head(decoder_output)
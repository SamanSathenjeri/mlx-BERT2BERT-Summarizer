import math
import mlx.core as mx
import mlx.nn as nn
import matplotlib.pyplot as plt

from BERTEncoder import BERT, BERT_Config, BERT_Attention, BERT_Embedding

def visualize_attention(attn_weights, input_tokens, output_tokens, layer=0, head=0, mode="cross"):
    attn = attn_weights[layer][0, head].tolist()  # (T_tgt, T_src)
    plt.figure(figsize=(12, 6))
    plt.imshow(attn, cmap="viridis")
    plt.xticks(ticks=range(len(input_tokens)), labels=input_tokens, rotation=90)
    plt.yticks(ticks=range(len(output_tokens)), labels=output_tokens)
    plt.xlabel("Input Tokens" if mode == "cross" else "Past Output Tokens")
    plt.ylabel("Generated Tokens")
    plt.title(f"{mode.capitalize()} Attention - Layer {layer}, Head {head}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

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

    def __call__(self, x, context, mask=None, return_attn=False):
        assert x.ndim == 3, f"x should be (B, T, E) but got shape {x.shape}"
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

        output = self.output(output)

        if return_attn:
            return output, attn_weights
        
        return output

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

    def __call__(self, x, encoder_hidden_states, self_mask=None, cross_mask=None, return_attn=False):
        if return_attn:
            self_attn_out, self_attn_weights = self.self_attn(x, mask=self_mask, return_attn=True)
            cross_out, cross_weights = self.cross_attn(
                self_attn_out, context=encoder_hidden_states, mask=cross_mask, return_attn=True
            )
            final = self.layer_norm1(cross_out)
            return final, self_attn_weights, cross_weights
        else:
            self_attn_out = self.self_attn(x, mask=self_mask, return_attn=False)
            cross_out = self.cross_attn(self_attn_out, context=encoder_hidden_states, mask=cross_mask)
            return self.layer_norm1(cross_out)

class BERTDecoder(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config
        self.embedding = BERT_Embedding(config.vocab_size, config.block_size, config.num_embd)
        self.blocks = [DecoderBlock(config) for _ in range(config.num_layers)]
        self.output_proj = nn.Linear(config.num_embd, config.vocab_size)

    def __call__(self, tokens: mx.array, encoder_hidden_states, decoder_pad_mask=None, encoder_pad_mask=None, return_attn=False):
        B, T = tokens.shape
        H = self.config.num_heads
        x = self.embedding(tokens)
        causal = causal_mask(T)[None, None, :, :]
        causal = mx.broadcast_to(causal, (B, H, T, T))
        S = encoder_hidden_states.shape[1]

        assert decoder_pad_mask.shape == (B, 1, 1, T), f"Got {decoder_pad_mask.shape}"
        assert encoder_pad_mask.shape == (B, 1, 1, S), f"Got {encoder_pad_mask.shape}"

        if decoder_pad_mask is not None:
            self_mask = causal * decoder_pad_mask
        else:
            self_mask = causal

        if encoder_pad_mask is not None:
            encoder_mask = encoder_pad_mask
        else:
            encoder_mask = None

        self_attns, cross_attns = [], []
        for block in self.blocks:
            if return_attn:
                x, self_attn, cross_attn = block(
                    x,
                    encoder_hidden_states,
                    self_mask=self_mask,
                    cross_mask=encoder_mask,
                    return_attn=True,
                )
                self_attns.append(self_attn)
                cross_attns.append(cross_attn)
            else:
                x = block(
                    x,
                    encoder_hidden_states,
                    self_mask=self_mask,
                    cross_mask=encoder_mask,
                )

        logits = self.output_proj(x)

        if return_attn:
            return logits, self_attns, cross_attns
        return logits

class BERT2BERT(nn.Module):
    def __init__(self, config: BERT_Config):
        super().__init__()
        self.config = config
        self.encoder = BERT(config, encoder_Flag=True)
        self.decoder = BERTDecoder(config)
        self.freeze_encoder_weights = True

    def __call__(self, encoder_input_ids, decoder_input_ids):
        encoder_pad_mask = make_padding_mask(encoder_input_ids)
        decoder_pad_mask = make_padding_mask(decoder_input_ids)

        encoder_output = self.encoder(encoder_input_ids, mask=encoder_pad_mask, return_embeddings=True)

        if self.freeze_encoder_weights:
            encoder_output = mx.stop_gradient(encoder_output)

        decoder_output = self.decoder(
            decoder_input_ids,
            encoder_output,
            decoder_pad_mask=decoder_pad_mask,
            encoder_pad_mask=encoder_pad_mask,
            return_attn=True
        )
        return decoder_output
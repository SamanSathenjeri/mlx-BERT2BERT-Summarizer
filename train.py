import os, math, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertTokenizerFast
from datasets import load_dataset
import numpy as np

from BERTEncoder import BERT, BERT_Config

def prepare_corpus(tokenizer, dataset_name="wikitext", subset="wikitext-2-raw-v1", split="train", block_size=128):
    """
    Tokenize with special tokens; pack into contiguous blocks; allow padding on the tail.
    """
    ds = load_dataset(dataset_name, subset, split=split)  # {'text': [...]}
    texts = [t for t in ds["text"] if t and not t.isspace()]

    # tokenize with special tokens; no filtering
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
        padding=False,
        max_length=block_size,
    )

    # flatten into one long stream of ids with [SEP] boundaries already present
    # ids = list(np.fromiter((tid for seq in enc["input_ids"] for tid in seq), dtype=np.int32))
    ids = [int(tid) for seq in enc["input_ids"] for tid in seq]

    # chunk into block_size windows
    chunks = []
    for i in range(0, len(ids), block_size):
        window = ids[i:i+block_size]
        if len(window) == 0: continue
        if len(window) < block_size:
            # pad last window
            window = window + [tokenizer.pad_token_id] * (block_size - len(window))
        chunks.append(window)

    # print(f"Type of pad_token_id: {type(tokenizer.pad_token_id)}")
    # print(f"Sample window: {chunks[0]}")
    # print(f"Type of first element: {type(chunks[0][0])}")
    arr = mx.array(chunks, dtype=mx.int32)  # (N, T)

    # attention mask: 1 for non-pad
    attn = (arr != tokenizer.pad_token_id).astype(mx.int32)
    return arr, attn

def dynamic_mlm_mask(batch_inputs, tokenizer, mlm_prob=0.15):
    """
    Apply BERT MLM 80/10/10 per BERT paper, excluding specials & pads.
    Returns (inputs_corrupted, labels) where labels=-100 for non-MLM positions.
    Shapes preserved.
    """
    inputs = batch_inputs
    B, T = inputs.shape
    labels = mx.full((B, T), vals=-100, dtype=mx.int32)

    # boolean masks
    pad_id = tokenizer.pad_token_id
    specials = set(tokenizer.all_special_ids)

    # allowed_to_mask: not pad and not special
    not_pad = (inputs != pad_id) if pad_id is not None else mx.ones_like(inputs, dtype=mx.bool_)
    special_mask = None
    for sid in specials:
        m = (inputs == sid)
        special_mask = m if special_mask is None else (special_mask | m)
    not_special = ~special_mask if special_mask is not None else mx.ones_like(inputs, dtype=mx.bool_)
    allowed = not_pad & not_special

    # sample positions
    prob = mx.random.uniform(shape=inputs.shape)
    final_mask = (prob < mlm_prob) & allowed

    # set labels for masked positions
    labels = mx.where(final_mask, inputs, labels)

    # 80% -> [MASK]
    # 10% -> random token (!= pad)
    # 10% -> unchanged
    r = mx.random.uniform(shape=inputs.shape)
    mask80 = final_mask & (r < 0.8)
    rand10 = final_mask & (r >= 0.8) & (r < 0.9)

    x = inputs
    # [MASK]
    x = mx.where(mask80, tokenizer.mask_token_id, x)

    # random tokens (avoid pad id)
    vocab_size = tokenizer.vocab_size
    if pad_id is None or pad_id < 0 or pad_id >= vocab_size:
        rand_ids = mx.random.randint(low=0, high=vocab_size, shape=inputs.shape, dtype=mx.int32)
    else:
        raw = mx.random.randint(low=0, high=vocab_size - 1, shape=inputs.shape, dtype=mx.int32)
        rand_ids = raw + (raw >= pad_id).astype(mx.int32)
    x = mx.where(rand10, rand_ids, x)

    return x, labels

def collate_batch(arr_ids, arr_attn, idxs):
    # simple gather by indices
    batch_inputs = arr_ids[idxs]
    batch_attn   = arr_attn[idxs]
    return batch_inputs, batch_attn

def param_groups_for_weight_decay(model, weight_decay=0.01):
    """
    Separate params into decay/non-decay like PyTorch best practice.
    """
    flat = dict(utils.tree_flatten(model.parameters()))
    # print(model.parameters())
    # print("\n\n*************************************************************\n\n")
    # print(flat)
    decay, nodecay = {}, {}
    for k, v in flat.items():
        # heuristics: no decay for LayerNorm scale/bias and bias terms
        name = k.lower()
        if any(s in name for s in ["layer_norm", "ln", "norm"]) or name.endswith(".bias"):
            nodecay[k] = v
        else:
            decay[k] = v
    return (
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    )

def cross_entropy_ignore_index(logits, targets, ignore_index=-100):
    """
    logits: (B, T, V), targets: (B, T) int
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape((B*T, V))
    targets_flat = targets.reshape((B*T,))
    try:
        losses = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=ignore_index)
    except TypeError:
        # older MLX fallback
        safe_targets = mx.where(targets_flat == ignore_index, mx.zeros_like(targets_flat), targets_flat)
        losses = nn.losses.cross_entropy(logits_flat, safe_targets, reduction='none')
        mask = (targets_flat != ignore_index).astype(losses.dtype)
        losses = losses * mask
        denom = mx.maximum(mask.sum(), mx.array(1, dtype=mask.dtype))
        return losses.sum() / denom

    valid = (targets_flat != ignore_index).astype(losses.dtype)
    denom = mx.maximum(valid.sum(), mx.array(1, dtype=valid.dtype))
    return (losses * valid).sum() / denom

# ---------- Train / Eval

def evaluate(model, arr_ids, arr_attn, batch_size, tokenizer):
    model.eval()
    num = arr_ids.shape[0]
    steps = max(1, num // batch_size)
    tot = 0.0
    for s in range(steps):
        start = s * batch_size
        end = start + batch_size
        idxs = slice(start, end)
        inp, attn = arr_ids[idxs], arr_attn[idxs]
        # dynamic masking on eval too (standard for MLM)
        x, labels = dynamic_mlm_mask(inp, tokenizer)
        with nn.stateful() as _:
            logits = model(x, mask=attn)
        loss = cross_entropy_ignore_index(logits, labels)
        tot += float(loss.item())
    return tot / steps

def train_loop(config_path="config.yaml"):
    config = BERT_Config.from_yaml(config_path) if hasattr(BERT_Config, "from_yaml") else BERT_Config.from_yaml()
    model = BERT(config)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "[PAD]"  # safety; bert-base-uncased already has PAD=0

    T = config.block_size
    train_ids, train_attn = prepare_corpus(tokenizer, split="train", block_size=T)
    val_ids,   val_attn   = prepare_corpus(tokenizer, split="validation", block_size=T)

    # Schedules
    batch_size = 64 if not hasattr(config, "batch_size") else config.batch_size
    epochs     = 5  if not hasattr(config, "epochs") else config.epochs
    base_lr    = 3e-4 if not hasattr(config, "learning_rate") else config.learning_rate
    warmup_pct = 0.06
    weight_decay = 0.01
    max_grad_norm = 1.0

    steps_per_epoch = train_ids.shape[0] // batch_size
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(warmup_pct * total_steps))

    # cosine with warmup
    def lr_schedule(step):
        step = mx.array(step, dtype=mx.float32)
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        # cosine decay to 10% of base_lr
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + mx.cos(math.pi * progress)))

    # optimizer with decoupled weight decay
    group_decay, group_nodecay = param_groups_for_weight_decay(model, weight_decay)
    opt = optim.AdamW(learning_rate=lr_schedule, betas=(0.9, 0.999))  # groups supported via update call
    # opt.init(model.trainable_parameters())

    # value_and_grad expects f(model, *args)
    def loss_fn(m, batch_inputs, batch_attn, tokenizer_obj):
        x, labels = dynamic_mlm_mask(batch_inputs, tokenizer_obj)
        logits = m(x, mask=batch_attn)
        return cross_entropy_ignore_index(logits, labels)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print("Starting training...")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        # shuffle indices
        perm = np.random.permutation(train_ids.shape[0])
        epoch_loss = 0.0
        for step_idx in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}"):
            idx = perm[step_idx*batch_size:(step_idx+1)*batch_size]
            idx = mx.array(idx, dtype=mx.int32)
            batch_inputs, batch_attn = train_ids[idx], train_attn[idx]

            # compute loss and grads
            loss, grads = loss_and_grad(model, batch_inputs, batch_attn, tokenizer)
            print("Model params:", model.parameters().keys())
            print("Grad keys:", grads.keys())
            # clip
            grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
            # # update with per-group weight decay
            # opt.update(group_decay, grads)      # decay group
            # opt.update(group_nodecay, grads)    # no-decay group
            opt.update(model, grads)
            # sync
            mx.eval(model.parameters(), opt.state, loss)
            epoch_loss += float(loss.item())

        avg_train = epoch_loss / steps_per_epoch
        avg_val = evaluate(model, val_ids, val_attn, batch_size, tokenizer)
        train_losses.append(avg_train); val_losses.append(avg_val)
        print(f"Epoch {epoch}: train {avg_train:.4f} | val {avg_val:.4f}")

    # save weights
    flat = dict(utils.tree_flatten(model.parameters()))
    meta = {"model_type":"BERT","framework":"MLX","desc":"MLX BERT MLM","saved_at": time.strftime("%Y-%m-%d")}
    os.makedirs("safetensors", exist_ok=True)
    mx.save_safetensors("safetensors/BERT_weights.safetensors", flat, metadata=meta)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BERT Pretraining Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    return train_losses, val_losses

if __name__ == "__main__":
    train_loop()

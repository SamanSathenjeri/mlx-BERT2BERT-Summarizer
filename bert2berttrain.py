import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from BERTEncoder import BERT_Config, BERT
from BERTDecoder import BERT2BERT

BATCH_SIZE = 8
SEQ_LEN = 64
NUM_EPOCHS = 3
LR = 1e-4
PAD_TOKEN_ID = 0
PRETRAINED_ENCODER_PATH = "./checkpoints/bert_encoder.safetensors"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"

def preprocess(example):
    text = example["text"]
    if not text.strip():
        return None
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="np")
    return {
        "encoder_input_ids": tokens["input_ids"][0],
        "decoder_input_ids": tokens["input_ids"][0][:-1],
        "labels": tokens["input_ids"][0][1:],
        "padding_mask": (tokens["attention_mask"][0][1:]).astype("float32"),
    }

raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train") # Use a larger split if possible
train_split_idx = int(len(raw_dataset) * 0.9)
raw_train_dataset = raw_dataset.select(range(train_split_idx))
raw_val_dataset = raw_dataset.select(range(train_split_idx, len(raw_dataset)))

train_dataset = [preprocess(e) for e in raw_train_dataset if preprocess(e) is not None]
val_dataset = [preprocess(e) for e in raw_val_dataset if preprocess(e) is not None]

def create_batches(data, batch_size):
    filtered_data = [d for d in data if d is not None]
    if not filtered_data:
        return []
    for i in range(0, len(filtered_data), batch_size):
        batch = filtered_data[i:i+batch_size]
        yield {
            k: mx.array(np.stack([sample[k] for sample in batch]))
            for k in batch[0]
        }

config = BERT_Config.from_yaml("./config.yaml")
model = BERT2BERT(config)
model.encoder.load_weights(PRETRAINED_ENCODER_PATH)

for p in model.encoder.parameters():
    p.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=LR)

def seq2seq_loss(logits, labels, padding_mask):
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    labels = labels.flatten()
    padding_mask = padding_mask.flatten()
    loss = mx.losses.cross_entropy(logits, labels, reduction="none")
    masked_loss = loss * padding_mask
    return mx.mean(masked_loss)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_train_loss = 0
    batches = create_batches(train_dataset, BATCH_SIZE)
    num_train_batches = 0
    for step, batch in enumerate(batches):
        num_train_batches += 1
        logits = model(batch["encoder_input_ids"], batch["decoder_input_ids"])
        loss = seq2seq_loss(logits, batch["labels"], batch["padding_mask"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += float(loss)

        if step % 50 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Train Loss: {float(loss):.4f}")

    avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
    train_losses.append(avg_train_loss)
    print(f">>> Epoch {epoch+1} completed, Avg Train Loss: {avg_train_loss:.4f}")

    epoch_val_loss = 0
    val_batches = create_batches(val_dataset, BATCH_SIZE)
    num_val_batches = 0
    for step, batch in enumerate(val_batches):
        num_val_batches += 1
        logits = model(batch["encoder_input_ids"], batch["decoder_input_ids"])
        loss = seq2seq_loss(logits, batch["labels"], batch["padding_mask"])
        epoch_val_loss += float(loss)

    avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
    val_losses.append(avg_val_loss)
    print(f">>> Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', marker='x')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- Save Decoder Weights ---
model.save_weights("bert2bert_decoder.safetensors")